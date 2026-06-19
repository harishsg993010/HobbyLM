"""Is the model-driven harness GENERALIZABLE ACROSS UIs?

Same architecture as computer_use.py, applied to WITHIN-APP interaction:
  * the MODEL decides the VERB (click / type_text / press_key)  -- model in the loop, not optional
  * DETERMINISTIC UIA grounding resolves the TARGET to a live accessibility-tree element, and the TEXT
    to type from the instruction (the parts the model drifts on).

Generality claim: every Windows app exposes a UIA tree, so the SAME harness enumerates whatever window
is focused and grounds to it -- no per-app code. We test it unchanged across Calculator (buttons) and
Notepad (text field + menus) and report verb-accuracy + target-resolution. Resolution-only: NO clicks.
"""
import sys, os, time, json
from urllib.request import Request, urlopen
import ui_agent_som as U

FC_TOOLS = [
    {"type": "function", "function": {"name": "click", "description": "Click a UI element.",
        "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}},
    {"type": "function", "function": {"name": "type_text", "description": "Type text into a field.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "press_key", "description": "Press a keyboard key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
]
VERBS = {"click", "type_text", "press_key"}


def model_verb(instr):
    """Model picks the verb (+ maybe a target/text). -> (verb|None, args)."""
    body = json.dumps({"model": "moe-omni-500m", "messages": [{"role": "user", "content": instr}],
                       "tools": FC_TOOLS, "tool_choice": "required", "max_tokens": 40, "temperature": 0}).encode()
    try:
        m = json.load(urlopen(Request(U.API, data=body, headers={"Content-Type": "application/json"}),
                              timeout=60))["choices"][0]["message"]
    except Exception:
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


def det_verb(instr):
    ql = instr.lower()
    if any(w in ql for w in ("type", "write", "enter ")):
        return "type_text"
    if any(w in ql for w in ("press", "hit ")):
        return "press_key"
    return "click"


def resolve(instr, els, win):
    """Model decides verb; deterministic grounds target/text. Returns (verb, resolution_str, src)."""
    mv, ma = model_verb(instr)
    verb = mv if mv in VERBS else det_verb(instr)
    src = "model" if mv in VERBS else f"recover({mv})"
    if verb == "type_text":
        text = U.extract_text(instr)
        field = U.find_field(win, instr)
        fname = (field[0] or "(main)") if field else "NONE"
        return verb, f"text={text!r} -> field {fname!r}", src
    if verb == "press_key":
        key = ma.get("key") or instr.split()[-1]
        return verb, f"key={key!r}", src
    # click: ground the target to a live element (model's target first, else the instruction)
    idx = U.match_element(ma.get("target", ""), els) if ma.get("target") else -1
    if idx < 0:
        idx = U.match_element(instr, els)
    return verb, (U.friendly(els[idx][0]) if idx >= 0 else "UNRESOLVED"), src


APPS = {
    "Calculator": ("calc", [
        ("click 7", "7"), ("click the plus button", "plus +"), ("click 5", "5"),
        ("click equals", "equals ="), ("clear the display", "clear c"),
    ]),
    "Notepad": ("notepad", [
        ("type Hello world", "Hello world"), ("click the File menu", "file"), ("click Edit", "edit"),
        ("type 'meeting at noon'", "meeting at noon"),
    ]),
}


def main():
    grand_v = grand_r = grand_n = 0
    for app, (launch, cases) in APPS.items():
        os.startfile(launch)
        time.sleep(1.5)
        win = U.get_window(app)
        if not win:
            print(f"\n## {app}: window not found, skipping"); continue
        try:
            win.SetActive()
        except Exception:
            pass
        time.sleep(0.4)
        els = U.enumerate_elements(win)
        print(f"\n## {app}  ({len(els)} UIA elements enumerated)")
        for instr, exp in cases:
            verb, res, src = resolve(instr, els, win)
            grand_n += 1
            want_verb = "type_text" if instr.lower().startswith(("type", "write")) else "click"
            vok = (verb == want_verb)
            # resolution correctness: any expected token shows up in the resolution
            rok = any(tok in res.lower() for tok in exp.lower().split())
            grand_v += vok; grand_r += rok
            print(f"  {instr:28s} {verb:10s}[{'v' if vok else 'x'}/{src:10s}] -> {res:28s} [{'OK' if rok else 'x'}]")
    print("-" * 92)
    print(f"TOTAL {grand_n}: verb-correct {grand_v}/{grand_n}, target-resolved {grand_r}/{grand_n}")


if __name__ == "__main__":
    main()
