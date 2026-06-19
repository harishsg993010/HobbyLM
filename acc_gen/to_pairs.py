#!/usr/bin/env python3
"""Convert the raw {query,tools,answers} computer-use data into the {prompt,completion,tools,answers}
single-shot pairs that moe-lab's ToolCallSFT / evaluate read — using the IDENTICAL serialization the
500M was tool-trained on (tool_data.flatten_tools + 'TOOLS: ...\\nUSER: ...\\nASSISTANT:' + ' '+answers).

    python to_pairs.py            # converts acc_computeruse.{train,val,test}.jsonl -> acc_{train,val,test}.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # moe-lab on path
from tool_data import flatten_tools  # noqa: E402


def convert(row):
    tools = flatten_tools(json.loads(row["tools"]))
    tools_json = json.dumps(tools, separators=(",", ":"))
    prompt = "TOOLS: " + tools_json + "\nUSER: " + row["query"] + "\nASSISTANT:"
    completion = " " + row["answers"]  # answers is already compact json.dumps([{name,arguments}])
    return {"prompt": prompt, "completion": completion, "tools": tools_json, "answers": row["answers"]}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for split in ("train", "val", "test"):
        src = os.path.join(here, f"acc_computeruse.{split}.jsonl")
        dst = os.path.join(here, f"acc_{split}.jsonl")
        n = 0
        with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                fo.write(json.dumps(convert(json.loads(line)), ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n} -> {dst}", file=sys.stderr)


if __name__ == "__main__":
    main()
