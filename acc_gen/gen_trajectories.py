#!/usr/bin/env python3
"""Generate multi-step PLANNING trajectories for v4, from real (and synthetic) accessibility trees.

For each screen, Gemini invents a few realistic multi-step GOALS, each as an ordered list of atomic
grounded actions. We expand every trajectory into per-step training rows in the planning format:

    SCREEN:<tree>

    GOAL: <goal>
    DONE: <atomic instr 1>; <atomic instr 2>
    NEXT:                              ->  [<next grounded action>]   (or [{"name":"finish",...}] at the end)

The model thus learns to DECOMPOSE the goal AND ground the next step, using the done-history as state.
Trained alongside v3's single-step grounding data, v4 keeps both modes (atomic-instruction grounding
when there's no GOAL line; planning when there is). Output: same {query,tools,answers} format, split 90/5/5.

    GEMINI_API_KEY=...  python gen_trajectories.py --trees real_trees.jsonl --per-tree 6 --split 90/5/5
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
from gen_accessibility import ACTIONS, validate_task, _norm, ClientPool, make_clients, MODEL
from gen_from_real import cap_tree, _serialize

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(*a, **k):
        class _N:
            n = 0
            def update(self, x=1): pass
            def close(self): pass
        return _N()

# v4 action vocabulary = the 12 grounding actions + `finish` (emitted when the goal is complete).
FINISH = {"name": "finish", "description": "Signal that the goal has been fully achieved; no more actions are needed.",
          "parameters": {}}
ACTIONS_PLAN = ACTIONS + [FINISH]
ACTIONS_PLAN_STR = json.dumps(ACTIONS_PLAN, separators=(",", ":"), ensure_ascii=False)
PLAN_ANAMES = sorted(a["name"] for a in ACTIONS_PLAN)

_PROMPT = """You are generating MULTI-STEP task trajectories to train a tiny on-device agent that controls a real Windows app via its accessibility tree.

Below is a REAL accessibility-tree snapshot of a window. Invent {k} realistic MULTI-STEP user GOALS achievable ON THIS screen, each broken into an ordered sequence of 2-6 ATOMIC steps.

{screen}

Return a JSON array of goals:
[
  {{"goal":"a natural high-level user goal",
    "steps":[
      {{"instruction":"atomic step phrased like a user command, e.g. 'click the seven button'",
        "action":{{"name":"click","arguments":{{"element":"Seven","control_type":"Button"}}}}}}
    ]}}
]

RULES:
- Each `goal` is something a real user would ask for that genuinely takes MULTIPLE steps on THIS screen (e.g. on a calculator 'calculate 7 plus 2'; on settings 'turn on dark mode and increase brightness'; on a form 'fill in the name and email then submit').
- Each step's `instruction` is ONE atomic action in plain user language.
- Each step's `action.name` is one of: {anames} (do NOT use `finish` here; it is added automatically).
- GROUNDING: for click/double_click/right_click/hover/type_text/select/set_value/drag, `arguments.element` MUST be the EXACT name of an element in the tree above, and `arguments.control_type` MUST match that element's [type]. For drag, `arguments.target` too. type_text needs `text`; set_value needs numeric `value`; press_key needs `key`; scroll needs `direction`.
- Steps must be in the correct ORDER to achieve the goal. Use elements that really exist above.
- Vary goals widely; make them concrete and realistic for this specific app.

Return ONLY the JSON array."""


def gen_traj_for_tree(pool, tree, rng, model, k, max_el=45):
    els = cap_tree(tree["elements"], max_el)
    screen = _serialize(tree["window"], els)
    name_to_types = {}
    for e in els:
        name_to_types.setdefault(_norm(e["name"]), set()).add(e["control_type"])

    prompt = _PROMPT.format(k=k, screen=screen, anames=", ".join(sorted(a["name"] for a in ACTIONS)))
    try:
        resp = pool.get().models.generate_content(
            model=model, contents=prompt,
            config={"temperature": rng.choice([0.7, 0.8, 0.9, 1.0]), "max_output_tokens": 8192,
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
        goals = json.loads(text.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(goals, list):
        return []

    rows = []
    for g in goals:
        if not isinstance(g, dict):
            continue
        goal = str(g.get("goal", "")).strip()
        steps = g.get("steps")
        if not goal or not isinstance(steps, list) or len(steps) < 2:
            continue
        # Validate every step grounds; keep the trajectory only if all steps are clean.
        cleaned_steps = []
        ok = True
        for st in steps:
            if not isinstance(st, dict):
                ok = False
                break
            cleaned = validate_task(st, name_to_types)
            instr = str(st.get("instruction", "")).strip()
            if cleaned is None or not instr:
                ok = False
                break
            cleaned_steps.append((instr, cleaned))
        if not ok or len(cleaned_steps) < 2:
            continue

        # Expand into per-step training rows: at step i, history = instructions[0..i), target = action[i].
        # A final row emits `finish`.
        for i in range(len(cleaned_steps) + 1):
            done = "; ".join(instr for instr, _ in cleaned_steps[:i])
            done_line = done if done else "nothing yet"
            query = f"{screen}\n\nGOAL: {goal}\nDONE: {done_line}\nNEXT:"
            if i < len(cleaned_steps):
                target = [cleaned_steps[i][1]]
            else:
                target = [{"name": "finish", "arguments": {}}]
            rows.append({
                "query": query,
                "tools": ACTIONS_PLAN_STR,
                "answers": json.dumps(target, separators=(",", ":"), ensure_ascii=False),
                "source": "synth-gemini-trajectory",
                "label": tree["label"], "goal": goal, "step": i,
                "action": target[0]["name"], "traj_len": len(cleaned_steps),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", default="real_trees.jsonl")
    ap.add_argument("--per-tree", type=int, default=6, help="Gemini calls per tree")
    ap.add_argument("--goals", type=int, default=5, help="goals requested per call")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="acc_traj.jsonl")
    ap.add_argument("--split", default="90/5/5")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    trees = [json.loads(l) for l in open(args.trees, encoding="utf-8")]
    pool = ClientPool(make_clients())
    rng = random.Random(args.seed)
    jobs = [(t, random.Random(rng.randint(0, 2**31))) for t in trees for _ in range(args.per_tree)]
    print(f"{len(trees)} trees x {args.per_tree} calls x {args.goals} goals", file=sys.stderr)

    rows, seen, lock = [], set(), threading.Lock()
    pbar = tqdm(total=len(jobs), desc="trees x calls", unit="call")

    def _one(job):
        t, jr = job
        return gen_traj_for_tree(pool, t, jr, args.model, args.goals)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for batch in ex.map(_one, jobs):
            for r in batch:
                key = r["label"] + "||" + _norm(r["goal"]) + f"||{r['step']}"
                with lock:
                    if key in seen:
                        continue
                    seen.add(key); rows.append(r)
            pbar.update(1)
    pbar.close()

    from collections import Counter
    print(f"\nGenerated {len(rows)} planning-step examples "
          f"(~{len(set((r['label'], r['goal']) for r in rows))} trajectories).", file=sys.stderr)
    print("by action:", dict(Counter(r["action"] for r in rows)), file=sys.stderr)

    if args.dry_run:
        for r in rows[:6]:
            print("\n" + "=" * 70)
            print(r["query"].split("\n\n")[-1])
            print("ANSWER:", r["answers"])
        return

    pcts = [float(x) for x in args.split.split("/")]
    srng = random.Random(args.seed)
    # Split by TRAJECTORY (keep all steps of a goal together) to avoid leakage.
    trajs = {}
    for r in rows:
        trajs.setdefault((r["label"], r["goal"]), []).append(r)
    keys = list(trajs)
    srng.shuffle(keys)
    n = len(keys); a = round(n * pcts[0] / 100); b = round(n * pcts[1] / 100)
    parts = {"train": keys[:a], "val": keys[a:a + b], "test": keys[a + b:]}
    base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    for sp, ks in parts.items():
        recs = [r for k in ks for r in trajs[k]]
        srng.shuffle(recs)
        with open(f"{base}.{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {sp}: {len(recs)} steps / {len(ks)} trajectories", file=sys.stderr)


if __name__ == "__main__":
    main()
