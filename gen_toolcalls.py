#!/usr/bin/env python3
"""Generate ENGLISH-ONLY general function-calling data with needle's recipe, for a stronger text-only
tool-use model. Loads needle's generator directly (bypassing the package __init__ that pulls in jax),
forces English, writes {query,tools,answers} + converts to {prompt,completion} single-shot pairs, split.

    GEMINI_API_KEY=...  python gen_toolcalls.py --num-samples 16000 --workers 16 --split 96/2/2
"""
import argparse
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ndlgen", os.path.join(HERE, "needle", "needle", "dataset", "generate.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
G.LANGUAGES = ["English"]  # English only (needle defaults to 25 languages)

sys.path.insert(0, HERE)
from tool_data import flatten_tools


def to_pair(row):
    tj = json.dumps(flatten_tools(json.loads(row["tools"])), separators=(",", ":"))
    return {"prompt": "TOOLS: " + tj + "\nUSER: " + row["query"] + "\nASSISTANT:",
            "completion": " " + row["answers"], "tools": tj, "answers": row["answers"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--out", default="toolcalls_en.jsonl")
    ap.add_argument("--split", default="96/2/2")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--bias-parallel", action="store_true",
                    help="bias call-types toward multi/parallel calls (to push BFCL parallel categories)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.bias_parallel:
        # Keep only the multi-call types (parallel) + a little single, so most examples have 2-4 calls.
        multi = [ct for ct in G.CALL_TYPES if ct[0] in ("multi", "multi_few_tools", "multi_long_values")]
        single = [ct for ct in G.CALL_TYPES if ct[0] == "single"][:1]
        G.CALL_TYPES = multi + single
        print(f"bias-parallel: CALL_TYPES -> {[c[0] for c in G.CALL_TYPES]}", file=sys.stderr)

    rows = G.generate_all(args.num_samples, workers=args.workers, batch_size=args.batch_size)
    from collections import Counter
    print(f"\nGenerated {len(rows)} examples.", file=sys.stderr)
    print("langs:", dict(Counter(r.get("language") for r in rows)), file=sys.stderr)
    print("call_types:", dict(Counter(r.get("call_type") for r in rows)), file=sys.stderr)
    print("num_tools:", dict(sorted(Counter(r.get("num_tools") for r in rows).items())), file=sys.stderr)

    if args.dry_run:
        for r in rows[:6]:
            print("\nQ:", r["query"][:100]); print("A:", r["answers"][:140])
        return

    pcts = [float(x) for x in args.split.split("/")]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows); a = round(n * pcts[0] / 100); b = round(n * pcts[1] / 100)
    parts = {"train": rows[:a], "val": rows[a:a + b], "test": rows[a + b:]}
    base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    for sp, rs in parts.items():
        with open(f"{base}_{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(to_pair(r), ensure_ascii=False) + "\n")
        print(f"  {sp}: {len(rs)} -> {base}_{sp}.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
