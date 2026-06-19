#!/usr/bin/env python3
"""Multi-step PLANNING via an observe->act loop over the hobby-chat API.

The 500M was trained single-shot (one action per screen), so we don't ask it for a whole plan at once
(that's out-of-distribution). Instead each step: snapshot the LIVE tree -> tell the model the goal +
what it has already done -> it emits ONE next action -> execute -> re-snapshot. The live screen is the
state (the model reads the updated display rather than tracking it internally — the thing that broke the
earlier agent). Loop until the model repeats/stalls or MAX_STEPS.

    python agent_loop.py --window Calculator --goal "calculate 7 plus 2 then equals" --execute --max-steps 8
"""
import argparse
import sys
import time

import uiautomation as auto
from ui_harness import snapshot, model_action, execute, get_window


def display_of(win):
    """Best-effort current-state string (Calculator display etc.) for logging."""
    try:
        d = win.TextControl(searchDepth=20, AutomationId="CalculatorResults")
        if d.Exists(0):
            return (d.Name or "").strip()
    except Exception:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--max-steps", type=int, default=8)
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
    last_sig = None
    print(f"GOAL: {args.goal}\n")
    for step in range(1, args.max_steps + 1):
        els, screen = snapshot(win)
        # Convey progress so the model doesn't repeat the first action (live display alone isn't enough).
        instr = args.goal
        if history:
            instr += " — done so far: " + ", ".join(history) + ". Next single action:"
        action = model_action(screen, instr)
        if action is None:
            print(f"[{step}] model produced no action — stopping"); break
        sig = (action["name"], str(action["arguments"].get("element", "")), str(action["arguments"].get("key", "")))
        a = action["arguments"]
        label = a.get("element") or a.get("key") or a.get("direction") or a.get("app_name") or ""
        print(f"[{step}] {action['name']}({label})   display={display_of(win)!r}")
        if sig == last_sig:
            print(f"     repeated action — stopping (likely done or stuck)"); break
        last_sig = sig
        res = execute(action, els, args.execute)
        print(f"     {'EXEC' if args.execute else 'DRY'}: {res}")
        if res.startswith(("UNRESOLVED", "FAILED")):
            print("     could not act — stopping"); break
        history.append(f"{action['name']} {label}".strip())
        time.sleep(0.6)

    print(f"\nFINAL display={display_of(win)!r}  ({len(history)} actions)")


if __name__ == "__main__":
    main()
