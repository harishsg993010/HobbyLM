"""End-to-end agentic task: 'open calculator and calculate 2 + 2'.

Spans both harnesses:
  1. the MODEL decides the action to open Calculator (computer_use.model_call -> open_app, executed),
  2. the in-app UIA harness drives the buttons: the expression is tokenized and each digit/operator is
     grounded to a live Calculator button via match_element and clicked, then the display is read.
"""
import sys, re, time
import uiautomation as auto
import ui_agent_som as U
import computer_use as CU

EXPR = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "2 + 2"


def open_calculator():
    instr = "open the calculator"
    name, args, src = CU.model_call(instr)
    tool, fargs, fixed = CU.repair(name, args, instr)
    cmd, status = CU.execute(tool, fargs, do_real=True)
    print(f"[1] ask model: {instr!r}")
    print(f"    model -> {name}({args}) [{src}] -> {tool}({fargs}) :: {status}  ({cmd})")
    return tool == "open_app"


def tokens(expr):
    out = []
    for ch in re.findall(r"\d|[+\-*/=]", expr):
        out.append(ch)
    if out and out[-1] != "=":
        out.append("=")
    return out


def click_token(tok, els):
    idx = U.match_element(tok, els)
    if idx < 0:
        print(f"    click {tok!r}: UNRESOLVED"); return False
    nm, box = els[idx]
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    import pyautogui
    pyautogui.click(cx, cy); time.sleep(0.3)
    print(f"    click {tok!r:3s} -> {U.friendly(nm)!r} @ ({cx},{cy})")
    return True


def read_display(win):
    found = [None]

    def walk(c, d=0):
        if d > 22 or found[0]:
            return
        for ch in c.GetChildren():
            try:
                nm = ch.Name or ""
                if nm.lower().startswith("display is"):
                    found[0] = nm; return
            except Exception:
                pass
            walk(ch, d + 1)
    walk(win)
    return found[0]


def main():
    if not open_calculator():
        print("model did not choose open_app; aborting"); return
    time.sleep(1.8)
    win = U.get_window("Calculator")
    if not win:
        print("[!] Calculator window not found"); return
    try:
        win.SetActive()
    except Exception:
        pass
    time.sleep(0.5)
    els = U.enumerate_elements(win)
    toks = tokens(EXPR)
    print(f"[2] calculate {EXPR!r}  -> buttons {toks}  ({len(els)} elements)")
    for t in toks:
        click_token(t, els)
    time.sleep(0.4)
    disp = read_display(win)
    print(f"[3] result: {disp!r}")
    m = re.search(r"display is\s*([-\d.,]+)", (disp or ""), re.I)
    print(f"    => {EXPR} = {m.group(1) if m else '??'}")


if __name__ == "__main__":
    main()
