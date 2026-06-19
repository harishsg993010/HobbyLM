"""Reliable agentic tool use for the omni model — the forced-call policy baked in.

When tools are provided and the user makes a request, we COMMIT to a tool call (like an API's
tool_choice="required"): the decode is seeded with `[` so the model emits a call instead of chatting,
then the name is snapped to a valid tool and the JSON repaired. After the tool result is supplied, the
model summarizes it. Works for a TEXT query or a SPOKEN query (Whisper features spliced at <speech>).

    call   = agent_tool_call(vlm, tok, dev, tools_json, query="...")               # or speech_features=...
    answer = agent_summarize(vlm, tok, dev, tools_json, call, result, query="...")  # or speech_features=...
"""
from __future__ import annotations

import json

import torch

from multimodal import SPEECH_TOKEN
from generate import GPT2_VALID, EOT


def _repair_and_snap(text, names):
    """Pull out the JSON array, balance brackets, snap each tool name to the schema (longest common prefix)."""
    s = text.strip()
    i = s.find("[")
    if i >= 0:
        s = s[i:]
    s = s + "}" * max(0, s.count("{") - s.count("}")) + "]" * max(0, s.count("[") - s.count("]"))
    try:
        calls = json.loads(s)
    except Exception:
        return text
    if isinstance(calls, dict):
        calls = [calls]
    if not isinstance(calls, list):
        return text
    nset = set(names)
    for c in calls:
        if isinstance(c, dict) and names and c.get("name") not in nset:
            cn = str(c.get("name", ""))
            c["name"] = max(names, key=lambda nm: len(_lcp(nm, cn)))
    return json.dumps(calls, separators=(",", ":"))


def _lcp(a, b):
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return a[:i]


def _banned(prev, k=3):
    if len(prev) < k:
        return []
    seen = {}
    for j in range(len(prev) - k + 1):
        seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
    return seen.get(tuple(prev[-(k - 1):]), [])


@torch.no_grad()
def _gen(vlm, tok, dev, ids, speech_features, max_new, rep):
    """Autoregressive greedy from token ids (SPEECH sentinel expanded if speech_features given)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if speech_features is not None:
            cur, _ = vlm.build_inputs_embeds(torch.tensor([ids], device=dev), speech_features=speech_features)
        else:
            cur = vlm.llm.embed(torch.tensor([ids], device=dev))
        outs = []
        for _ in range(max_new):
            lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
            lg[:, GPT2_VALID:] = -float("inf")
            if rep and outs:
                u = torch.tensor(sorted(set(outs)), device=dev)
                v = lg[0, u]
                lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
            t = int(lg.argmax(-1).item())
            if t == EOT:
                break
            outs.append(t)
            cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
    return tok.decode(outs).strip()


def _ctx(tok, tools_json, query, speech, assistant_tail):
    """Build token ids: TOOLS + (text query | <speech>) + assistant_tail."""
    ids = tok.encode_ordinary(f"TOOLS: {tools_json}\n")
    if speech:
        ids = ids + [SPEECH_TOKEN]
    else:
        ids = ids + tok.encode_ordinary(f"USER: {query}")
    return ids + tok.encode_ordinary(assistant_tail)


def agent_tool_call(vlm, tok, dev, tools_json, query=None, speech_features=None, max_new=48):
    """Forced-call policy: commit to a tool call, then snap the name to the schema + repair JSON."""
    speech = speech_features is not None
    ids = _ctx(tok, tools_json, query, speech, "\nASSISTANT: [")     # seed with '[' -> a call, not prose
    out = _gen(vlm, tok, dev, ids, speech_features, max_new, rep=False)
    try:
        names = [t["name"] for t in json.loads(tools_json) if isinstance(t, dict) and t.get("name")]
    except Exception:
        names = []
    return _repair_and_snap("[" + out, names)


def agent_summarize(vlm, tok, dev, tools_json, call, tool_result, query=None, speech_features=None, max_new=64):
    """Feed the tool result back and produce the natural-language answer."""
    speech = speech_features is not None
    if not isinstance(tool_result, str):
        tool_result = json.dumps(tool_result, separators=(",", ":"))
    ids = _ctx(tok, tools_json, query, speech, f"\nASSISTANT: {call}\nTOOL: {tool_result}\nASSISTANT:")
    return _gen(vlm, tok, dev, ids, speech_features, max_new, rep=True)
