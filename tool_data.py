"""Tool-use (function-calling) SFT data from nvidia/Nemotron-Agentic-v1 — Needle-style single-shot.

We take each conversation's FIRST assistant turn that emits tool_calls and build one example:
  prompt     = "TOOLS: [<schemas>]\n<serialized context>\nASSISTANT:"
  completion = ' [{"name": ..., "arguments": {...}}]'
Loss is on the completion only. Decoder-only (GPT-2 tiktoken), so tools/calls are plain JSON text.
`prep_pairs` does the extraction + length filter ONCE (writing prompt/completion jsonl); `ToolCallSFT`
just reads those prebuilt pairs at train time.
"""
from __future__ import annotations

import json

import tiktoken
import torch
from torch.utils.data import Dataset

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256
IGNORE_INDEX = -1


def flatten_tools(tools):
    """Nemotron OpenAI tools ({type,function:{name,description,parameters}}) -> [{name,description,parameters}]."""
    out = []
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        out.append({"name": fn.get("name"), "description": fn.get("description", "") or "",
                    "parameters": fn.get("parameters", {}) or {}})
    return out


def extract_singleshot(row):
    """(prompt, completion, tools_json, answers_json) for the first assistant tool-call turn, or None."""
    msgs = row.get("messages", [])
    tools = flatten_tools(row.get("tools", []))
    if not tools:
        return None
    ctx = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", {}) or {}
                args = f.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                calls.append({"name": f.get("name"), "arguments": args})
            tools_json = json.dumps(tools, separators=(",", ":"))
            parts = ["TOOLS: " + tools_json]
            for cm in ctx:
                role = (cm.get("role", "") or "").upper()
                content = cm.get("content", "") or ""
                if content:
                    parts.append(f"{role}: {content}")
            prompt = "\n".join(parts) + "\nASSISTANT:"
            answers_json = json.dumps(calls, separators=(",", ":"))
            return prompt, " " + answers_json, tools_json, answers_json
        ctx.append(m)
    return None


def _calls_to_pair(prompt, calls, tools_json):
    ans = json.dumps([{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in calls],
                     separators=(",", ":"))
    return prompt, " " + ans, tools_json, ans


def extract_bitagent(row):
    """BitAgent/tool_calling: conversation (user / 'tool call' {name,arguments} / assistant) + tools (param
    spec under 'arguments'). First 'tool call' turn is the target."""
    conv = row.get("conversation"); tools_raw = row.get("tools")
    if isinstance(conv, str):
        conv = json.loads(conv)
    if isinstance(tools_raw, str):
        tools_raw = json.loads(tools_raw)
    tools = [{"name": t.get("name"), "description": t.get("description", "") or "",
              "parameters": t.get("arguments") or t.get("parameters") or {}}
             for t in (tools_raw or []) if isinstance(t, dict)]
    if not tools:
        return None
    ctx = []
    for m in conv:
        if m.get("role") == "tool call":
            c = m["content"]
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except Exception:
                    return None
            calls = c if isinstance(c, list) else [c]
            tools_json = json.dumps(tools, separators=(",", ":"))
            parts = ["TOOLS: " + tools_json]
            for cm in ctx:
                content = cm.get("content", "")
                if isinstance(content, str) and content:
                    parts.append(f"{(cm.get('role') or '').upper()}: {content}")
            return _calls_to_pair("\n".join(parts) + "\nASSISTANT:", calls, tools_json)
        ctx.append(m)
    return None


def extract_interstellar(row):
    """interstellarninja/tool-calls-singleturn: ShareGPT conversations + OpenAI tool strings; the first gpt
    turn carries <tool_call>{...}</tool_call> blocks."""
    import re
    tools = []
    for ts in (row.get("tools") or []):
        try:
            t = json.loads(ts) if isinstance(ts, str) else ts
        except Exception:
            continue
        fn = t.get("function", t) if isinstance(t, dict) else {}
        params = fn.get("parameters", {})
        if isinstance(params, dict) and isinstance(params.get("properties"), str):
            try:
                params = {**params, "properties": json.loads(params["properties"])}
            except Exception:
                pass
        tools.append({"name": fn.get("name"), "description": fn.get("description", "") or "", "parameters": params})
    if not tools:
        return None
    query, calls = None, None
    for m in row.get("conversations", []):
        f, v = m.get("from"), m.get("value", "")
        if f == "human":
            query = v
        elif f == "gpt" and "<tool_call>" in v:
            calls = []
            for mt in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", v, re.S):
                try:
                    c = json.loads(mt.group(1))
                    calls.append({"name": c.get("name"), "arguments": c.get("arguments", {})})
                except Exception:
                    pass
            break
    if not query or not calls:
        return None
    tools_json = json.dumps(tools, separators=(",", ":"))
    return _calls_to_pair("TOOLS: " + tools_json + "\nUSER: " + query + "\nASSISTANT:", calls, tools_json)


EXTRACTORS = {"nemotron": extract_singleshot, "bitagent": extract_bitagent, "interstellar": extract_interstellar}


def _calls_json(tool_calls):
    out = []
    for c in tool_calls:
        f = c.get("function", c) or {}
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        out.append({"name": f.get("name"), "arguments": args})
    return json.dumps(out, separators=(",", ":"))


def extract_trajectory(row, source="nemotron"):
    """FULL multi-turn trajectory -> list of (text, is_loss) segments. Loss is on EVERY assistant turn
    (tool calls AND text answers); tool results stay in context. Parallel calls are kept (all calls in a
    turn -> one [..] target). This is the agentic-loop training signal (observe results, chain, abstain)."""
    segs = []
    if source == "nemotron":
        tools = flatten_tools(row.get("tools", []))
        if not tools:
            return None
        segs.append(("TOOLS: " + json.dumps(tools, separators=(",", ":")) + "\n", 0))
        msgs = row.get("messages", [])
    elif source == "bitagent":
        conv = row.get("conversation"); tr = row.get("tools")
        if isinstance(conv, str):
            conv = json.loads(conv)
        if isinstance(tr, str):
            tr = json.loads(tr)
        tools = [{"name": t.get("name"), "description": t.get("description", "") or "",
                  "parameters": t.get("arguments") or t.get("parameters") or {}} for t in (tr or []) if isinstance(t, dict)]
        if not tools:
            return None
        segs.append(("TOOLS: " + json.dumps(tools, separators=(",", ":")) + "\n", 0))
        msgs = []
        for m in conv:
            r = m.get("role"); c = m.get("content")
            if r == "tool call":
                cc = json.loads(c) if isinstance(c, str) else c
                calls = cc if isinstance(cc, list) else [cc]
                msgs.append({"role": "assistant", "tool_calls": [{"function": x} for x in calls]})
            elif r == "assistant":
                msgs.append({"role": "assistant", "content": c if isinstance(c, str) else ""})
            elif r in ("user", "system"):
                msgs.append({"role": r, "content": c if isinstance(c, str) else ""})
            elif r in ("tool", "tool response"):
                msgs.append({"role": "tool", "content": c if isinstance(c, str) else json.dumps(c)})
    else:
        return None

    has_assistant = False
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if not isinstance(content, str):               # tool results are dicts/lists -> stringify
            content = json.dumps(content, separators=(",", ":")) if content is not None else ""
        if role == "assistant":
            tc = m.get("tool_calls")
            target = _calls_json(tc) if tc else content
            if not target.strip():
                continue
            segs.append(("ASSISTANT:", 0))
            segs.append((" " + target, 1))             # loss on the target (+ EOT appended at tokenize time)
            has_assistant = True
        elif role == "system":
            if content:
                segs.append(("SYSTEM: " + content[:1200] + "\n", 0))
        elif role == "user":
            segs.append(("USER: " + content + "\n", 0))
        elif role == "tool":
            segs.append(("TOOL: " + content + "\n", 0))
    return segs if has_assistant else None


def extract_chat(messages, max_sys=1500):
    """Plain chat conversation (messages [{role,content}]) -> trajectory segments, loss on assistant
    turns only. Same SYSTEM:/USER:/ASSISTANT: format the engine + hobby-chat use. No tools."""
    segs, has_assistant = [], False
    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role == "assistant":
            if not content.strip():
                continue
            segs.append(("ASSISTANT:", 0))
            segs.append((" " + content, 1))            # loss on the answer (+ EOT at tokenize time)
            has_assistant = True
        elif role == "system":
            if content.strip():
                segs.append(("SYSTEM: " + content[:max_sys] + "\n", 0))
        elif role == "user":
            segs.append(("USER: " + content + "\n", 0))
    return segs if has_assistant else None


def prep_chat(rows_iter, max_len=2048):
    """Iterable of message-lists -> {segments} examples that fit max_len (>=1 assistant turn)."""
    out, total, none, too_long = [], 0, 0, 0
    for messages in rows_iter:
        total += 1
        try:
            segs = extract_chat(messages)
        except Exception:
            segs = None
        if not segs:
            none += 1
            continue
        n = sum(len(ENC.encode_ordinary(t)) + (1 if isl else 0) for t, isl in segs)
        if n > max_len:
            too_long += 1
            continue
        out.append({"segments": segs})
    return out, {"total": total, "none": none, "too_long": too_long, "kept": len(out)}


def prep_trajectories(rows, source, max_len=2048):
    """rows -> list of {segments:[...]} multi-turn examples that fit max_len (>=1 assistant turn)."""
    out, total, none, too_long = [], 0, 0, 0
    for row in rows:
        total += 1
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except Exception:
                continue
        try:
            segs = extract_trajectory(row, source)
        except Exception:
            segs = None
        if not segs:
            none += 1
            continue
        n = sum(len(ENC.encode_ordinary(t)) + (1 if isl else 0) for t, isl in segs)
        if n > max_len:
            too_long += 1
            continue
        out.append({"segments": segs})
    return out, {"total": total, "none": none, "too_long": too_long, "kept": len(out)}


class TrajectorySFT(Dataset):
    """Multi-turn trajectories; next-token loss only on assistant-turn tokens (+ their EOT)."""
    def __init__(self, path: str, max_len: int = 2048):
        self.data = []
        for p in path.split(","):
            with open(p.strip()) as f:
                self.data += [json.loads(l) for l in f]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _segments(self, ex):
        if "segments" in ex:
            return ex["segments"]
        # single-shot pair {prompt(ends with 'ASSISTANT:'), completion} -> a 1-turn must-call trajectory
        p = ex["prompt"]
        head = p[:-len("ASSISTANT:")] if p.endswith("ASSISTANT:") else p
        return [(head, 0), ("ASSISTANT:", 0), (ex["completion"], 1)]

    def __getitem__(self, i):
        ids, loss = [], []
        for text, is_loss in self._segments(self.data[i]):
            t = ENC.encode_ordinary(text)
            if is_loss:
                t = t + [EOT]                          # end each assistant turn with EOT
            ids += t
            loss += [1 if is_loss else 0] * len(t)
        ids, loss = ids[:self.max_len], loss[:self.max_len]
        L = len(ids)
        inp = torch.tensor(ids[:-1], dtype=torch.long)
        tgt = torch.tensor([ids[k + 1] if loss[k + 1] else IGNORE_INDEX for k in range(L - 1)], dtype=torch.long)
        return inp, tgt


def prep_rows(rows, extractor, max_len=2048, max_prompt=1900):
    """Generic: iterable of dict rows -> list of {prompt, completion, tools, answers} that fit max_len."""
    pairs, total, no_call, too_long = [], 0, 0, 0
    for row in rows:
        total += 1
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except Exception:
                continue
        try:
            ex = extractor(row)
        except Exception:
            ex = None
        if ex is None:
            no_call += 1
            continue
        prompt, completion, tools_json, answers_json = ex
        pt = len(ENC.encode_ordinary(prompt))
        ct = len(ENC.encode_ordinary(completion)) + 1
        if pt > max_prompt or pt + ct > max_len:
            too_long += 1
            continue
        pairs.append({"prompt": prompt, "completion": completion, "tools": tools_json, "answers": answers_json})
    return pairs, {"total": total, "no_call": no_call, "too_long": too_long, "kept": len(pairs)}


def prep_pairs(raw_lines, max_len=2048, max_prompt=1900):
    """Back-compat: Nemotron jsonl lines -> pairs."""
    return prep_rows(raw_lines, extract_singleshot, max_len, max_prompt)


import re

_VAL_RE = re.compile(r':\s*("(?:[^"\\]|\\.)*"|-?\d[\d.eE+-]*|true|false|null|\[[^\]]*\])')
_NAME_RE = re.compile(r'"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
_KEY_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:')


def completion_char_weights(comp: str, w_name=3.0, w_value=2.0, w_key=1.5):
    """Per-character loss weights over a tool-call completion: emphasize tool NAMES (3x), argument
    VALUES (2x) and keys (1.5x) over JSON scaffolding (1x) — Needle's weighting scheme."""
    w = [1.0] * len(comp)
    for m in _KEY_RE.finditer(comp):
        for i in range(m.start(1), m.end(1)):
            w[i] = w_key
    for m in _VAL_RE.finditer(comp):
        for i in range(m.start(1), m.end(1)):
            w[i] = w_value
    for m in _NAME_RE.finditer(comp):          # name value wins (applied last)
        for i in range(m.start(1), m.end(1)):
            w[i] = w_name
    return w


class ToolCallSFT(Dataset):
    """Reads prebuilt {prompt, completion} pairs; next-token loss masked to the completion (tool call).
    weighted=True also returns per-token loss weights (name/value/key emphasis)."""
    def __init__(self, path: str, max_len: int = 2048, weighted: bool = False):
        self.data = []                                  # path may be comma-separated -> combine sources
        for p in path.split(","):
            with open(p.strip()) as f:
                self.data += [json.loads(l) for l in f]
        self.max_len = max_len
        self.weighted = weighted

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        p = ENC.encode_ordinary(ex["prompt"])
        comp = ex["completion"]
        c = ENC.encode_ordinary(comp) + [EOT]
        logical = (p + c)[:self.max_len]
        np_ = len(p)
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor([logical[k + 1] if (k + 1) >= np_ else IGNORE_INDEX
                            for k in range(len(logical) - 1)], dtype=torch.long)
        if not self.weighted:
            return ids, tgt
        # weight aligned to TARGETS: target[k]=logical[k+1]; completion targets start at k+1>=np_.
        cw = completion_char_weights(comp)
        tok_w = []                                  # weight per completion token (the EOT gets 1.0)
        pos = 0
        for tid in ENC.encode_ordinary(comp):
            s = ENC.decode([tid])
            seg = cw[pos:pos + len(s)]
            tok_w.append(max(seg) if seg else 1.0)
            pos += len(s)
        tok_w.append(1.0)                           # EOT
        weights = torch.zeros(len(logical) - 1, dtype=torch.float32)
        for k in range(len(logical) - 1):
            if (k + 1) >= np_:
                j = (k + 1) - np_
                if j < len(tok_w):
                    weights[k] = tok_w[j]
        return ids, tgt, weights


def tool_collate(batch):
    ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
    return pad_ids, pad_tgt


def tool_collate_weighted(batch):
    ids, tgts, ws = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    pad_w = torch.zeros(B, L, dtype=torch.float32)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
        pad_w[b, :n] = ws[b]
    return pad_ids, pad_tgt, pad_w
