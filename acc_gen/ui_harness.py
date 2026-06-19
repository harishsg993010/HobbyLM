#!/usr/bin/env python3
"""Model-driven computer-use harness over the hobby-chat OpenAI API.

Loop: snapshot the live Windows UI Automation (accessibility) tree of a target window ->
serialize it EXACTLY like the training data (`SCREEN:\\n[ControlType] "Name" (state)`) ->
ask the 500M computer-use model (via hobby-chat /v1/chat/completions, tools=the 12 actions,
tool_choice=required) for ONE grounded action -> resolve the model's {element,control_type}
back to a live element via fuzzy match -> execute with pyautogui (or just print, in --dry-run).

The MODEL picks the verb + names the target element; the deterministic match_element layer
grounds that name to on-screen coordinates (the workhorse validated earlier). Nothing about
which action/element to use is hardcoded.

    # safe: resolve + print, no clicks
    python ui_harness.py --window Calculator --instr "click the equals button"
    # actually do it
    python ui_harness.py --window Calculator --instr "click 7" --execute

Run hobby-chat first:  HOBBYLM_CHAT_MODEL=...computeruse-hobbylm.gguf  hobby-chat.exe
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import uiautomation as auto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_accessibility import ACTIONS  # the fixed 12-action vocabulary (flat-param schema)

API = "http://127.0.0.1:11250/v1/chat/completions"

# UIA ControlTypeName -> the short control_type names used in the training data.
def short_ctype(ctn):
    return ctn[:-7] if ctn.endswith("Control") else ctn

# Broad capture (matches harvest_real.py): keep interactive controls AND context/chrome, like a
# real raw UI-Automation snapshot. The model (v3) was trained on real trees capped to ~45 elements
# (interactive-first) with chrome/context kept, so we capture the same way and cap to fit context —
# NO aggressive chrome stripping needed anymore.
KEEP = {"Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "List", "ListItem",
        "MenuItem", "Tab", "TabItem", "Slider", "Spinner", "Hyperlink", "TreeItem",
        "SplitButton", "ToggleButton", "Text", "Image", "Group", "Document", "ProgressBar"}
# Interactive types are prioritized when capping the tree to the context budget.
INTERACTIVE = {"Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "List", "ListItem",
               "MenuItem", "Tab", "TabItem", "Slider", "Spinner", "Hyperlink", "TreeItem",
               "SplitButton", "ToggleButton"}
MAX_ELEMENTS = 45  # cap to fit the model's 2048-token context (matches training)


def el_state(c, ctype):
    try:
        if not c.IsEnabled:
            return "disabled"
        if ctype in ("CheckBox", "RadioButton") and c.GetTogglePattern():
            return "checked" if c.GetTogglePattern().ToggleState == 1 else "unchecked"
    except Exception:
        pass
    try:
        rv = c.GetRangeValuePattern()
        if rv:
            return str(int(rv.Value)) if float(rv.Value).is_integer() else str(rv.Value)
    except Exception:
        pass
    return "enabled"


def snapshot(win, cap=160):
    """Walk the window -> list of {ctype,name,state,box}, plus serialized SCREEN string.
    Broad raw capture (interactive + context + chrome), then cap to MAX_ELEMENTS prioritizing
    interactive controls — exactly how the v3 training trees were built."""
    els, seen, seen_names = [], set(), set()

    def walk(c, d=0):
        if d > 32 or len(els) >= cap:
            return
        for ch in c.GetChildren():
            try:
                ctype = short_ctype(ch.ControlTypeName)
                nm = (ch.Name or ch.AutomationId or "").strip()
                r = ch.BoundingRectangle
                box = (r.left, r.top, r.right, r.bottom)
                vis = (r.right - r.left) > 3 and (r.bottom - r.top) > 3 and not ch.IsOffscreen
                nl = nm.lower()
                if ctype in KEEP and nm and vis and box not in seen and (ctype, nl) not in seen_names:
                    seen.add(box); seen_names.add((ctype, nl))
                    els.append({"ctype": ctype, "name": nm, "state": el_state(ch, ctype), "box": box})
            except Exception:
                pass
            walk(ch, d + 1)

    walk(win)
    # Cap to the context budget, interactive controls first, preserving original order (like cap_tree).
    if len(els) > MAX_ELEMENTS:
        inter = [e for e in els if e["ctype"] in INTERACTIVE]
        other = [e for e in els if e["ctype"] not in INTERACTIVE]
        keep = inter[:MAX_ELEMENTS]
        if len(keep) < MAX_ELEMENTS:
            keep += other[:MAX_ELEMENTS - len(keep)]
        keep_ids = {id(e) for e in keep}
        els = [e for e in els if id(e) in keep_ids]
    title = (win.Name or "Window").strip()
    lines = [f'[Window] "{title}"']
    for e in els:
        st = f"  ({e['state']})" if e["state"] else ""
        lines.append(f'[{e["ctype"]}] "{e["name"]}"{st}')
    return els, "SCREEN:\n" + "\n".join(lines)


# ---- fuzzy grounding: model's element name -> live element index ----
_STOP = {"the", "a", "an", "of", "to", "button", "field", "box", "the"}


def _keys(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w and w not in _STOP}


def match_element(name, ctype, els):
    """Best element whose name tokens overlap `name`; control_type match is a tiebreak bonus."""
    want = _keys(name)
    best, score = -1, 0
    for i, e in enumerate(els):
        ek = _keys(e["name"])
        s = sum(len(k) + 2 for k in ek if k in want)
        if e["name"].strip().lower() == name.strip().lower():
            s += 20  # exact name wins
        if ctype and e["ctype"].lower() == ctype.lower():
            s += 3
        if s > score:
            best, score = i, s
    return best if score >= 3 else -1


def center(box):
    return (box[0] + box[2]) // 2, (box[1] + box[3]) // 2


# ---- ask the model for one grounded action ----
def model_action(screen, instruction, seed=1234):
    payload = {
        "model": "moe-omni-500m",
        "messages": [{"role": "user", "content": f"{screen}\n\n{instruction}"}],
        "tools": [{"type": "function", "function": a} for a in ACTIONS],
        "tool_choice": "required",
        "temperature": 0,
        "seed": seed,
    }
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None
    fn = tcs[0]["function"]
    args = fn["arguments"]
    if isinstance(args, str):
        args = json.loads(args)
    return {"name": fn["name"], "arguments": args}


# ---- execute (or describe) a grounded action ----
def execute(action, els, do_real):
    import pyautogui
    pyautogui.FAILSAFE = True
    name = action["name"]
    a = action["arguments"]
    el = a.get("element")

    def resolve(nm, ct):
        idx = match_element(nm or "", ct or "", els)
        return (idx, els[idx]) if idx >= 0 else (-1, None)

    if name in ("click", "double_click", "right_click", "hover"):
        idx, e = resolve(el, a.get("control_type"))
        if idx < 0:
            return f"UNRESOLVED element {el!r}"
        cx, cy = center(e["box"])
        desc = f'{name} -> [{e["ctype"]}] "{e["name"]}" @ ({cx},{cy})'
        if do_real:
            {"click": pyautogui.click, "double_click": pyautogui.doubleClick,
             "right_click": pyautogui.rightClick, "hover": pyautogui.moveTo}[name](cx, cy)
            time.sleep(0.4)
        return desc

    if name == "type_text":
        idx, e = resolve(el, a.get("control_type"))
        text = a.get("text", "")
        if idx < 0:
            return f"UNRESOLVED field {el!r}"
        cx, cy = center(e["box"])
        desc = f'type {text!r} into [{e["ctype"]}] "{e["name"]}" @ ({cx},{cy})'
        if do_real:
            pyautogui.click(cx, cy); time.sleep(0.2)
            pyautogui.typewrite(text, interval=0.02)
        return desc

    if name == "press_key":
        key = a.get("key", "")
        desc = f"press_key {key!r}"
        if do_real:
            parts = [p.strip().lower() for p in re.split(r"[+\-]", key) if p.strip()]
            (pyautogui.hotkey(*parts) if len(parts) > 1 else pyautogui.press(parts[0]))
            time.sleep(0.3)
        return desc

    if name == "scroll":
        d = a.get("direction", "down")
        amt = -600 if d == "down" else 600 if d == "up" else 0
        desc = f"scroll {d}"
        if do_real and amt:
            pyautogui.scroll(amt); time.sleep(0.3)
        return desc

    if name == "drag":
        i1, e1 = resolve(el, a.get("control_type"))
        i2, e2 = resolve(a.get("target"), None)
        if i1 < 0 or i2 < 0:
            return f"UNRESOLVED drag {el!r}->{a.get('target')!r}"
        x1, y1 = center(e1["box"]); x2, y2 = center(e2["box"])
        desc = f'drag "{e1["name"]}" ({x1},{y1}) -> "{e2["name"]}" ({x2},{y2})'
        if do_real:
            pyautogui.moveTo(x1, y1); pyautogui.dragTo(x2, y2, duration=0.5)
        return desc

    if name == "select":
        idx, e = resolve(el, a.get("control_type"))
        val = a.get("value")
        if idx < 0:
            return f"UNRESOLVED select {el!r}"
        cx, cy = center(e["box"])
        desc = f'select {val!r} in [{e["ctype"]}] "{e["name"]}"'
        if do_real:
            pyautogui.click(cx, cy); time.sleep(0.4)  # open the dropdown; value pick left to a 2nd snapshot
        return desc

    if name == "set_value":
        idx, e = resolve(el, a.get("control_type"))
        if idx < 0:
            return f"UNRESOLVED set_value {el!r}"
        cx, cy = center(e["box"])
        desc = f'set_value [{e["ctype"]}] "{e["name"]}" = {a.get("value")}'
        if do_real and e["ctype"] in ("Edit", "Spinner"):
            pyautogui.click(cx, cy); pyautogui.hotkey("ctrl", "a")
            pyautogui.typewrite(str(a.get("value")), interval=0.02)
        return desc

    if name == "open_app":
        return f'open_app {a.get("app_name")!r} (not auto-launched in harness)'
    if name == "wait":
        s = float(a.get("seconds", 1))
        if do_real:
            time.sleep(min(s, 5))
        return f"wait {s}s"
    return f"(unhandled action {name})"


def get_window(name):
    if name == "--foreground":
        import ctypes
        h = ctypes.windll.user32.GetForegroundWindow()
        return auto.ControlFromHandle(h)
    w = auto.WindowControl(searchDepth=1, SubName=name)
    return w if w.Exists(3) else None


def run_one(win, instruction, do_real):
    els, screen = snapshot(win)
    action = model_action(screen, instruction)
    print(f'\nINSTR: {instruction}')
    if action is None:
        print("  model returned no tool call"); return
    print(f"  MODEL: {json.dumps(action)}")
    print(f"  {'EXEC' if do_real else 'DRY '}: {execute(action, els, do_real)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, help="window title substring, or --foreground")
    ap.add_argument("--instr", action="append", help="instruction (repeatable)")
    ap.add_argument("--execute", action="store_true", help="actually perform actions (default: dry-run)")
    ap.add_argument("--show-tree", action="store_true", help="print the serialized SCREEN tree")
    args = ap.parse_args()

    win = get_window(args.window)
    if win is None:
        print(f"window {args.window!r} not found", file=sys.stderr); sys.exit(1)
    try:
        win.SetFocus()
    except Exception:
        pass
    time.sleep(0.3)

    if args.show_tree:
        _, screen = snapshot(win)
        print(screen)
    for instr in (args.instr or []):
        run_one(win, instr, args.execute)


if __name__ == "__main__":
    main()
