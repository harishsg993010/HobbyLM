#!/usr/bin/env python3
"""Two-tier planning agent: a PLANNER decomposes a goal into atomic steps; the local 500M GROUNDS+acts.

The 500M can't decompose a goal itself (it parrots the whole goal as one element). So we split the job:
a capable planner (Gemini here) looks at the GOAL + the live accessibility tree + what's been done, and
emits ONE atomic natural-language instruction (e.g. "click the seven button") or "DONE". The local 500M
then does what it's good at — ground that atomic instruction to a real element and execute it. Re-snapshot,
repeat. This keeps grounding/execution fully local; only the planning step is delegated.

    GEMINI_API_KEY=...  python agent_planned.py --window Calculator --goal "calculate 7 plus 2" --execute
"""
import argparse
import os
import re
import sys
import time

from google import genai
from ui_harness import snapshot, model_action, execute, get_window
from agent_loop import display_of

PLANNER_MODEL = "gemini-3.1-flash-lite-preview"

_PLAN_PROMPT = """You are guiding a small on-device agent to control a Windows app via its accessibility tree.

GOAL: {goal}

CURRENT SCREEN (live accessibility tree):
{screen}

STEPS ALREADY DONE: {history}

Output the SINGLE next ATOMIC UI instruction to make progress toward the goal — phrased like a user command
the agent can ground to ONE element, e.g. "click the seven button", "type hello into the search box",
"press the equals button". The instruction must target an element visible in the screen above. If the GOAL
is already fully achieved (check the screen — e.g. the display already shows the answer), output exactly DONE.

Output ONLY the instruction text (or DONE). No quotes, no explanation."""


def plan_next(client, goal, screen, history):
    prompt = _PLAN_PROMPT.format(goal=goal, screen=screen,
                                 history=("; ".join(history) if history else "none"))
    r = client.models.generate_content(model=PLANNER_MODEL, contents=prompt,
                                        config={"temperature": 0.2, "max_output_tokens": 40})
    return (r.text or "").strip().strip('"').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("set GEMINI_API_KEY", file=sys.stderr); sys.exit(1)
    client = genai.Client(api_key=key)

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
        instr = plan_next(client, args.goal, screen, history)
        if re.match(r"^\s*done\b", instr, re.I):
            print(f"[{step}] planner: DONE  display={display_of(win)!r}"); break
        print(f"[{step}] plan: {instr!r}   display={display_of(win)!r}")
        action = model_action(screen, instr)              # local 500M grounds the atomic instruction
        if action is None:
            print("     500M produced no action — stopping"); break
        a = action["arguments"]
        label = a.get("element") or a.get("key") or a.get("direction") or ""
        res = execute(action, els, args.execute)
        print(f"     500M: {action['name']}({label}) -> {'EXEC' if args.execute else 'DRY'}: {res}")
        if res.startswith(("UNRESOLVED", "FAILED")):
            print("     could not act — stopping"); break
        history.append(instr)
        time.sleep(0.7)

    print(f"\nFINAL display={display_of(win)!r}  ({len(history)} steps)")


if __name__ == "__main__":
    main()
