"""A/B: prose target-naming vs structured function-calling for computer-use grounding.

METHOD A (current): model names the target in prose ("The element is the ..."), harness fuzzy-matches.
METHOD B (new):     model emits a structured tool call click({"element":..}) / type_text / press_key;
                    harness grounds the structured arg to the real UIA element.

Both share the SAME UIA enumeration + fuzzy match (ui_agent_som). We compare which RESOLVES to the
correct element. No clicks — resolution only.
"""
import sys, json, time
from urllib.request import Request, urlopen
import uiautomation as auto
import ui_agent_som as U

FC_TOOLS = [
    {"type": "function", "function": {"name": "click",
        "description": "Click a UI element on the screen.",
        "parameters": {"type": "object", "properties": {"element": {"type": "string"}}, "required": ["element"]}}},
    {"type": "function", "function": {"name": "type_text",
        "description": "Type text into a text field.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "field": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "press_key",
        "description": "Press a keyboard key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
]


def call_fc(task):
    body = json.dumps({"model": "moe-omni-500m", "messages": [{"role": "user", "content": task}],
                       "tools": FC_TOOLS, "tool_choice": "required", "max_tokens": 40, "temperature": 0}).encode()
    req = Request(U.API, data=body, headers={"Content-Type": "application/json"})
    m = json.load(urlopen(req, timeout=180))["choices"][0]["message"]
    tcs = m.get("tool_calls")
    if not tcs:
        return None, {}, (m.get("content") or "")
    f = tcs[0]["function"]
    try:
        args = json.loads(f.get("arguments") or "{}")
    except Exception:
        args = {}
    return f["name"], args, json.dumps(tcs[0]["function"])


def name_of(idx, els):
    return U.friendly(els[idx][0]) if idx is not None and idx >= 0 else None


def prose(task, els):
    """Returns (model_only_resolution, full_resolution_with_fallback)."""
    target = U.ask_target(task)
    mo = name_of(U.match_element(target, els), els)         # what the model's prose grounds to
    full = mo or name_of(U.match_element(task, els), els)   # + deterministic instruction fallback
    return mo, full, f"prose={target!r}"


def fc(task, els):
    name, args, raw = call_fc(task)
    mo = None
    if name == "click":
        mo = name_of(U.match_element(args.get("element", ""), els), els)
    # SAME deterministic fallback as prose, regardless of which tool the model chose
    full = mo or name_of(U.match_element(task, els), els)
    return mo, full, f"{name}({args})"


def main():
    win = U.get_window("Calculator")
    if not win:
        print("open Calculator first"); return
    win.SetActive(); time.sleep(0.3)
    els = U.enumerate_elements(win)
    cases = [("Click 7", "7"), ("Click 9", "9"), ("Click the plus button", "+ plus"),
             ("Click equals", "= equals"), ("Click 0", "0"),
             ("Press the divide button", "/ divide"), ("Tap backspace", "Backspace")]
    score = {"A_model": 0, "A_full": 0, "B_model": 0, "B_full": 0, "n": 0}
    for instr, exp in cases:
        if not exp:
            continue
        score["n"] += 1
        amo, afull, ad = prose(instr, els)
        bmo, bfull, bd = fc(instr, els)
        ok = lambda r: bool(r and exp in str(r))
        score["A_model"] += ok(amo); score["A_full"] += ok(afull)
        score["B_model"] += ok(bmo); score["B_full"] += ok(bfull)
        print(f"{instr:26s} exp={exp:10s}")
        print(f"   PROSE  model={str(amo):12s}{'OK' if ok(amo) else 'x':>3}  full={str(afull):12s}{'OK' if ok(afull) else 'x':>3}   ({ad[:34]})")
        print(f"   FUNC   model={str(bmo):12s}{'OK' if ok(bmo) else 'x':>3}  full={str(bfull):12s}{'OK' if ok(bfull) else 'x':>3}   ({bd[:34]})")
    n = score["n"]
    print("-" * 78)
    print(f"MODEL-ONLY contribution:  prose {score['A_model']}/{n}   func-call {score['B_model']}/{n}")
    print(f"WITH deterministic fallback: prose {score['A_full']}/{n}   func-call {score['B_full']}/{n}")


if __name__ == "__main__":
    main()
