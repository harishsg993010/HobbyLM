"""GENUINELY model-driven UI agent: the MODEL plans every step from the UIA state. Nothing about the
button sequence is hardcoded.

Loop:  observe (UIA: live buttons + current display)  ->  THINK (model picks the next button via a
click() tool call, or done())  ->  act (UIA clicks it)  ->  repeat.

The model decides WHICH button and WHEN to stop. The harness only: enumerates the UIA tree (perception),
grounds the model's chosen label to a real element (match_element), clicks it, and reads the display. If
the model picks wrong / stalls, you SEE it -- no deterministic plan rescues it.
"""
import sys, re, time, json
from urllib.request import Request, urlopen
import ui_agent_som as U

TASK = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "calculate 2 + 2"
MAX_STEPS = 8

TOOLS = [
    {"type": "function", "function": {"name": "click", "description": "Click one calculator button.",
        "parameters": {"type": "object", "properties": {"button": {"type": "string"}}, "required": ["button"]}}},
    {"type": "function", "function": {"name": "done", "description": "The calculation is finished.",
        "parameters": {"type": "object", "properties": {}}}},
]


def ask_model(task, display, pressed):
    """Give the model the live state; it returns the next action (model PLANS here)."""
    hist = ", ".join(pressed) if pressed else "nothing yet"
    user = (f"You are operating a calculator to: {task}. "
            f"Buttons you have pressed so far: {hist}. "
            f"Call click() with the single next button to press, or done() if finished.")
    body = json.dumps({"model": "moe-omni-500m", "messages": [{"role": "user", "content": user}],
                       "tools": TOOLS, "tool_choice": "required", "max_tokens": 30, "temperature": 0}).encode()
    try:
        m = json.load(urlopen(Request(U.API, data=body, headers={"Content-Type": "application/json"}),
                              timeout=60))["choices"][0]["message"]
    except Exception as e:
        return None, {}
    tcs = m.get("tool_calls")
    if not tcs:
        return None, {}
    f = tcs[0]["function"]
    try:
        a = json.loads(f.get("arguments") or "{}")
    except Exception:
        a = {}
    return f["name"], (a if isinstance(a, dict) else {})


def display_of(win):
    found = [None]

    def walk(c, d=0):
        if d > 22 or found[0]:
            return
        for ch in c.GetChildren():
            try:
                nm = ch.Name or ""
                mm = re.search(r"display is\s*(.+)$", nm, re.I)
                if mm:
                    found[0] = mm.group(1).strip()
                    return
            except Exception:
                pass
            walk(ch, d + 1)
    walk(win)
    return found[0] or "0"


def click_label(label, els):
    idx = U.match_element(label, els)
    if idx < 0:
        return None
    nm, box = els[idx]
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    import pyautogui
    pyautogui.click(cx, cy); time.sleep(0.3)
    return U.friendly(nm)


def main():
    win = U.get_window("Calculator")
    if not win:
        import os
        os.startfile("calc"); time.sleep(1.8)
        win = U.get_window("Calculator")
    if not win:
        print("Calculator not open"); return
    try:
        win.SetActive()
    except Exception:
        pass
    time.sleep(0.5)
    import pyautogui
    pyautogui.press("escape"); time.sleep(0.3)        # clear any stale state -> display 0

    pressed = []
    print(f"GOAL: {TASK}\n")
    for step in range(1, MAX_STEPS + 1):
        els = U.enumerate_elements(win)
        disp = display_of(win)
        verb, args = ask_model(TASK, disp, pressed)
        if verb == "done":
            print(f"  step {step}: display={disp:>4} | model -> done()"); break
        if verb != "click":
            print(f"  step {step}: display={disp:>4} | model -> {verb} (no usable action) STOP"); break
        label = args.get("button", "")
        clicked = click_label(label, els)
        print(f"  step {step}: display={disp:>4} | model -> click({label!r}) -> "
              f"{clicked or 'UNRESOLVED'}")
        if clicked is None:
            break
        pressed.append(clicked)
        if "=" in clicked or "equal" in clicked.lower():
            break
    time.sleep(0.4)
    print(f"\nFINAL display: {display_of(win)}   (model made {len(pressed)} clicks: {pressed})")


if __name__ == "__main__":
    main()
