"""Needle-protocol tool-call eval on the NVIDIA agentic format, through the LIVE serving path.

Streams nvidia/Nemotron-Agentic-v1 :: tool_calling, rebuilds each row as a single-shot OpenAI request
(tools + system/user), hits the local hobby-chat API (tool_choice=required), and scores predictions vs
ground truth with tool_eval.score_tool_calls (the exact Needle metrics).

NOTE: this samples FRESH rows (skips ahead), so some may overlap the training set -> read the numbers as
an IN-DISTRIBUTION FORMAT CEILING, not held-out generalization. For the true held-out number, run
`modal run modal_tools.py --action eval` against the volume's tools_val.jsonl.
"""
import sys, json
from urllib.request import Request, urlopen
from huggingface_hub import HfFileSystem
from tool_data import flatten_tools
from tool_eval import score_tool_calls

# stream the raw JSONL line-by-line (range reads) — avoids datasets' pyarrow schema inference (content
# is sometimes str, sometimes object) and the whole-file MemoryError.
HF_PATH = "datasets/nvidia/Nemotron-Agentic-v1/data/tool_calling.jsonl"


def stream_rows():
    fs = HfFileSystem()
    with fs.open(HF_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

API = "http://127.0.0.1:11250/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SKIP = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # stream from the start (skip-ahead is slow)
MAXLEN = 4000        # skip giant contexts to keep gens fast


def parse_row(row):
    """-> (tools_flat, ctx_messages, ref_calls) for the first assistant tool-call turn, else None."""
    tools = flatten_tools(row.get("tools", []))
    if not tools:
        return None
    ctx = []
    for m in row.get("messages", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = c.get("function", {}) or {}
                a = f.get("arguments", {})
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        pass
                calls.append({"name": f.get("name"), "arguments": a})
            return tools, ctx, calls
        ctx.append(m)
    return None


def call_api(messages, tools):
    oai = [{"type": "function", "function": t} for t in tools]
    body = json.dumps({"model": "moe-omni-500m", "messages": messages, "tools": oai,
                       "tool_choice": "required", "max_tokens": 160, "temperature": 0}).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    m = json.load(urlopen(req, timeout=180))["choices"][0]["message"]
    tcs = m.get("tool_calls")
    if not tcs:
        return (m.get("content") or "").strip()
    out = []
    for tc in tcs:
        f = tc["function"]
        try:
            a = json.loads(f.get("arguments") or "{}")
        except Exception:
            a = {}
        out.append({"name": f["name"], "arguments": a})
    return json.dumps(out, separators=(",", ":"))


def main():
    refs, preds, tools_col = [], [], []
    used = 0
    for i, row in enumerate(stream_rows()):
        if i < SKIP:
            continue
        p = parse_row(row)
        if not p:
            continue
        tools, ctx, calls = p
        messages = []
        for m in ctx:
            if m.get("role") not in ("system", "user"):
                continue
            c = m.get("content")
            c = c if isinstance(c, str) else (json.dumps(c) if c else "")
            if c:
                messages.append({"role": m["role"], "content": c})
        if not any(m["role"] == "user" for m in messages):
            continue
        if sum(len(m["content"]) for m in messages) > MAXLEN:
            continue
        ref = json.dumps(calls, separators=(",", ":"))
        tj = json.dumps(tools, separators=(",", ":"))
        try:
            pred = call_api(messages, tools)
        except Exception as e:
            print(f"  (api err {type(e).__name__}) skip", flush=True)
            continue
        refs.append(ref); preds.append(pred); tools_col.append(tj)
        used += 1
        if used <= 10:
            print(f"[{used}] REF  {ref[:90]}\n     PRED {pred[:90]}", flush=True)
        elif used % 20 == 0:
            print(f"  ...{used}/{N}", flush=True)
        if used >= N:
            break

    m = score_tool_calls(refs, preds, tools_col)
    print("=" * 72)
    print(f"N={len(refs)}  nvidia/Nemotron-Agentic-v1::tool_calling  (single-shot, live API)")
    for k, v in m.items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
