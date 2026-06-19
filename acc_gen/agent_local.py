#!/usr/bin/env python3
"""Fully-LOCAL planning agent (v4). One local model plans + grounds + executes — no cloud planner.

v4 was trained on planning trajectories, so it does the whole loop itself: each step we snapshot the live
tree and ask it (planning format: GOAL + DONE-history -> next action over the 13-action vocab incl `finish`)
for the SINGLE next grounded action. Ground it, execute, re-snapshot, repeat until it emits `finish`.

    python agent_local.py --window Calculator --goal "calculate 8 minus 5" --execute
"""
import argparse
import json
import sys
import time
import urllib.request

from ui_harness import snapshot, execute, get_window
from agent_loop import display_of

API = "http://127.0.0.1:11250/v1/chat/completions"

# 13-action vocabulary used in planning (12 grounding actions + finish), matching v4 training.
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from gen_trajectories import ACTIONS_PLAN  # noqa: E402


def instr_for(action):
    """Reconstruct the trained-style atomic instruction phrasing for the DONE-history (the model was
    trained on natural phrasings like 'click the Eight button', not terse 'click Eight')."""
    n = action["name"]
    a = action["arguments"]
    el = a.get("element", "")
    if n in ("click", "double_click", "right_click", "hover"):
        verb = {"click": "click", "double_click": "double-click", "right_click": "right-click", "hover": "hover over"}[n]
        return f"{verb} the {el} button"
    if n == "type_text":
        return f"type {a.get('text','')} into the {el} field"
    if n == "press_key":
        return f"press {a.get('key','')}"
    if n == "scroll":
        return f"scroll {a.get('direction','')}"
    if n == "select":
        return f"select {a.get('value','')} in {el}"
    if n == "set_value":
        return f"set {el} to {a.get('value','')}"
    return f"{n} {el}".strip()


def plan_step(screen, goal, history):
    done = "; ".join(history) if history else "nothing yet"
    content = f"{screen}\n\nGOAL: {goal}\nDONE: {done}\nNEXT:"
    payload = {
        "model": "moe-omni-500m",
        "messages": [{"role": "user", "content": content}],
        "tools": [{"type": "function", "function": a} for a in ACTIONS_PLAN],
        "tool_choice": "required",
        "temperature": 0, "seed": 1234,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    tcs = resp["choices"][0]["message"].get("tool_calls") or []
    if not tcs:
        return None
    fn = tcs[0]["function"]
    args = fn["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    return {"name": fn["name"], "arguments": args}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    win = get_window(args.window)
    if win is None:
        print(f"window {args.window!r} not found", file=sys.stderr); sys.exit(1)
    try:
        win.SetActive()
    except Exception:
        pass
    time.sleep(0.3)

    history = []
    print(f"GOAL: {args.goal}\n")
    for step in range(1, args.max_steps + 1):
        els, screen = snapshot(win)
        action = plan_step(screen, args.goal, history)
        if action is None:
            print(f"[{step}] no action — stopping"); break
        if action["name"] == "finish":
            print(f"[{step}] finish   display={display_of(win)!r}"); break
        a = action["arguments"]
        label = a.get("element") or a.get("key") or a.get("direction") or ""
        print(f"[{step}] {action['name']}({label})   display={display_of(win)!r}")
        if args.execute:
            try:
                win.SetActive(); win.SetTopmost(True)  # raise target above hobby-chat so clicks land on it
            except Exception:
                pass
            time.sleep(0.25)
        res = execute(action, els, args.execute)
        print(f"     {'EXEC' if args.execute else 'DRY'}: {res}")
        if res.startswith(("UNRESOLVED", "FAILED")):
            print("     could not act — stopping"); break
        history.append(instr_for(action))  # trained-style phrasing for the DONE-history
        time.sleep(0.6)

    print(f"\nFINAL display={display_of(win)!r}  ({len(history)} actions)")


if __name__ == "__main__":
    main()
