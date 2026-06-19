#!/usr/bin/env python3
"""Generate grounded computer-use tasks for REAL harvested accessibility trees.

Reads real_trees.jsonl (from harvest_real.py) and, for each REAL tree, asks Gemini to write diverse
user instructions + the correct grounded action — the tree is GIVEN (not invented), so the model trains
on the exact raw, cluttered distribution it sees at inference (chrome, decorative Text/Group, disabled
items, near-identical clusters). Output is the same {query,tools,answers} single-shot format, split 90/5/5.

    GEMINI_API_KEY=...  python gen_from_real.py --rounds 4 --tasks 16 --split 90/5/5
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from google import genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_accessibility import (ACTIONS_STR, ACTION_NAMES, validate_task, _norm,
                               ClientPool, make_clients, MODEL, _FOCUS_ACTIONS, _FOCUS_W, _FOCUS_HINT)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*a, **k):
        class _N:
            n = 0
            def update(self, x=1): pass
            def close(self): pass
        return _N()

_PROMPT = """You are generating training data for a tiny on-device model that controls a real Windows PC via the ACCESSIBILITY TREE (UI Automation), not screenshots.

Below is a REAL accessibility-tree snapshot of a live window. Generate {n} diverse user instructions for THIS EXACT screen, each paired with the single correct grounded action.

{screen}

RULES:
- `action.name` MUST be one of: {anames}.
- {emphasis}
- GROUNDING IS CRITICAL: for click/double_click/right_click/hover/type_text/select/set_value/drag, `arguments.element` MUST be the EXACT `name` of an element listed above — copy it verbatim (exact case/words). `arguments.control_type` MUST equal that element's bracketed [type]. For `drag`, `arguments.target` must also be an exact element name. NEVER invent an element that is not in the tree.
- Prefer REAL actionable controls: Button, Hyperlink, ListItem, MenuItem, CheckBox, RadioButton, Tab/TabItem, Edit, ComboBox, Slider, TreeItem. Do NOT target decorative [Text]/[Image]/[Group] containers, and do NOT target window chrome (Minimize/Maximize/Close/app icon) unless the instruction is genuinely about closing/minimizing.
- This tree is CLUTTERED and contains near-identical siblings; when you target one, make the instruction name WHICH one unambiguously (use its exact label).
- type_text.text comes from the user's words; set_value.value is numeric (Slider/Spinner/Edit only); press_key.key like "Enter","Tab","Ctrl+S","Delete"; scroll.direction up/down/left/right.
- Instructions sound like a real user talking to an assistant: terse, conversational, or indirect ("it's too quiet" -> raise a volume slider). Vary widely; never repeat an intent.
- Numbers are JSON numbers (75 not "75"); never emit partial actions.

Return ONLY a JSON array, nothing else:
[{{"instruction":"...","action":{{"name":"click","arguments":{{"element":"...","control_type":"Button"}}}}}}]"""


_INTERACTIVE = {"Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "List", "ListItem",
                "MenuItem", "Tab", "TabItem", "Slider", "Spinner", "Hyperlink", "TreeItem",
                "SplitButton", "ToggleButton"}


def cap_tree(elements, max_el):
    """Cap a real tree to fit the 2048-token context: keep all interactive controls (the action
    targets) first, then fill remaining budget with context/chrome, preserving original order so
    the serialized screen still reads like a real top-to-bottom snapshot."""
    if len(elements) <= max_el:
        return elements
    inter = [e for e in elements if e["control_type"] in _INTERACTIVE]
    other = [e for e in elements if e["control_type"] not in _INTERACTIVE]
    keep = inter[:max_el]
    if len(keep) < max_el:
        keep += other[:max_el - len(keep)]
    keep_ids = {id(e) for e in keep}
    return [e for e in elements if id(e) in keep_ids]


def _serialize(window, els):
    lines = [f'[Window] "{window}"']
    for e in els:
        st = f"  ({e['state']})" if e.get("state") else ""
        lines.append(f'[{e["control_type"]}] "{e["name"]}"{st}')
    return "SCREEN:\n" + "\n".join(lines)


def gen_for_tree(pool, tree, rng, model, n_tasks, max_el=45):
    els = cap_tree(tree["elements"], max_el)
    screen = _serialize(tree["window"], els)
    focus = rng.choices(_FOCUS_ACTIONS, weights=_FOCUS_W, k=1)[0]
    if focus == "_mix":
        emphasis = "Use a natural mix across all action types."
    else:
        hint = _FOCUS_HINT.get(focus)
        hs = f" (only if the screen supports it: {hint})" if hint else ""
        emphasis = f"Favor the `{focus}` action where it fits this screen{hs}; let the rest vary naturally."
    prompt = _PROMPT.format(n=n_tasks, screen=screen,
                            anames=", ".join(sorted(ACTION_NAMES)), emphasis=emphasis)
    # name -> control types present in THIS (capped) real tree
    name_to_types = {}
    for e in els:
        name_to_types.setdefault(_norm(e["name"]), set()).add(e["control_type"])

    temp = rng.choice([0.7, 0.8, 0.9, 1.0, 1.1])
    try:
        resp = pool.get().models.generate_content(
            model=model, contents=prompt,
            config={"temperature": temp, "max_output_tokens": 8192,
                    "response_mime_type": "application/json"})
    except Exception as e:
        print(f"  gemini error [{tree['label']}]: {str(e)[:100]}", file=sys.stderr)
        return []
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        tasks = json.loads(text.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(tasks, list):
        return []

    out = []
    for t in tasks:
        cleaned = validate_task(t, name_to_types)
        if cleaned is None:
            continue
        instr = str(t["instruction"]).strip()
        out.append({
            "query": f"{screen}\n\n{instr}",
            "tools": ACTIONS_STR,
            "answers": json.dumps([cleaned], separators=(",", ":"), ensure_ascii=False),
            "source": "synth-gemini-realtree",
            "label": tree["label"], "window": tree["window"],
            "action": cleaned["name"], "n_elements": len(els),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", default="real_trees.jsonl")
    ap.add_argument("--rounds", type=int, default=4, help="Gemini calls per tree (different focus each)")
    ap.add_argument("--tasks", type=int, default=16, help="tasks requested per call")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="acc_real.jsonl")
    ap.add_argument("--split", default="90/5/5")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    trees = [json.loads(l) for l in open(args.trees, encoding="utf-8")]
    print(f"{len(trees)} real trees | {args.rounds} rounds x {args.tasks} tasks each", file=sys.stderr)
    pool = ClientPool(make_clients())
    rng = random.Random(args.seed)
    jobs = [(tree, random.Random(rng.randint(0, 2**31))) for tree in trees for _ in range(args.rounds)]

    rows, seen, lock = [], set(), threading.Lock()
    pbar = tqdm(total=len(jobs), desc="trees x rounds", unit="call")

    def _one(job):
        tree, jrng = job
        return gen_for_tree(pool, tree, jrng, args.model if hasattr(args, "model") else MODEL, args.tasks)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for batch in ex.map(_one, jobs):
            for r in batch:
                key = r["label"] + "||" + _norm(r["query"].split("\n\n")[-1])
                with lock:
                    if key in seen:
                        continue
                    seen.add(key); rows.append(r)
            pbar.update(1)
    pbar.close()

    from collections import Counter
    print(f"\nGenerated {len(rows)} real-grounded examples.", file=sys.stderr)
    print("by action:", dict(Counter(r["action"] for r in rows)), file=sys.stderr)
    print("by app:", dict(Counter(r["label"] for r in rows)), file=sys.stderr)

    # stratified 90/5/5 by action
    pcts = [float(x) for x in args.split.split("/")]
    srng = random.Random(args.seed)
    by = {}
    for r in rows:
        by.setdefault(r["action"], []).append(r)
    train, val, test = [], [], []
    for recs in by.values():
        srng.shuffle(recs)
        n = len(recs); a = round(n * pcts[0] / 100); b = round(n * pcts[1] / 100)
        train += recs[:a]; val += recs[a:a + b]; test += recs[a + b:]
    for s in (train, val, test):
        srng.shuffle(s)
    base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    for name, recs in [(f"{base}.train.jsonl", train), (f"{base}.val.jsonl", val), (f"{base}.test.jsonl", test)]:
        with open(name, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"split -> train {len(train)} / val {len(val)} / test {len(test)}", file=sys.stderr)


if __name__ == "__main__":
    main()
