"""General accessibility-tree UI agent for the moe-omni-500m API — works on ANY Windows app with UI
Automation, not just Calculator.

The weak-model insight (from Set-of-Mark / a11y-grounding research): don't ask the 500M to predict
pixels or read marks. Instead:
  1. Enumerate the REAL clickable elements of the target window via Windows UI Automation — exact names
     + bounding boxes (ground truth from the OS), on-screen + enabled only.
  2. The model's ONLY job: NAME the target the instruction refers to (it echoes it; handles indirect
     phrasing like "log in" -> "Sign in"). No coordinates, no list to copy, no JSON.
  3. The harness fuzzy-matches that name to a UIA element (with a deterministic instruction-text
     fallback) and clicks the element's EXACT center -> pixel-perfect, model-independent.
  4. Re-enumerates after each click so multi-step tasks track the changing UI.

Usage:
  python ui_agent_som.py "Calculator" "Click 7" "Click plus" "Click 9" "Click equals" --execute
  python ui_agent_som.py --foreground "Click File"
  python ui_agent_som.py --list                      # list top-level windows
"""
import sys, json, time, re, ctypes
from ctypes import wintypes
from urllib.request import Request, urlopen
import uiautomation as auto

API = "http://127.0.0.1:11250/v1/chat/completions"
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

# interactive control types worth clicking, across typical Windows apps
CLICK_TYPES = {"ButtonControl", "SplitButtonControl", "MenuItemControl", "ListItemControl",
               "TabItemControl", "TreeItemControl", "CheckBoxControl", "RadioButtonControl",
               "HyperlinkControl", "ComboBoxControl", "EditControl", "DataItemControl"}
CHROME = ("minimize", "maximize", "close", "restore", "system menu")
# generic words that shouldn't drive a match
STOP = {"button", "the", "key", "press", "click", "tap", "number", "a", "an", "to", "on", "of",
        "for", "in", "open", "item", "menu", "icon", "select", "go", "this", "and", "or", "please"}
# normalize a few common symbol/number names so they match what users actually say
FRIENDLY = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
            "seven": "7", "eight": "8", "nine": "9", "plus": "+ plus", "minus": "- minus",
            "multiply by": "* multiply", "divide by": "/ divide", "equals": "= equals",
            "decimal separator": ". decimal"}
SYMS = set("+-*/=%")


def friendly(nm):
    return FRIENDLY.get(nm.strip().lower(), nm.strip())


def list_windows():
    out = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                r = wintypes.RECT(); user32.GetWindowRect(hwnd, ctypes.byref(r))
                if r.right - r.left > 200 and r.bottom - r.top > 120:
                    out.append(buf.value)
        return True

    user32.EnumWindows(cb, 0)
    return out


def get_window(name):
    if name == "--foreground":
        hwnd = user32.GetForegroundWindow()
        return auto.ControlFromHandle(hwnd)
    w = auto.WindowControl(searchDepth=1, SubName=name)
    return w if w.Exists(2) else None


def _collect(root, els, seen, cap):
    def walk(c, d=0):
        if d > 28 or len(els) >= cap:
            return
        for ch in c.GetChildren():
            try:
                if ch.ControlTypeName in CLICK_TYPES and ch.IsEnabled and not ch.IsOffscreen:
                    nm = (ch.Name or ch.AutomationId or "").strip()
                    r = ch.BoundingRectangle
                    box = (r.left, r.top, r.right, r.bottom)
                    if (nm and (r.right - r.left) > 3 and (r.bottom - r.top) > 3
                            and not any(k in nm.lower() for k in CHROME) and box not in seen):
                        seen.add(box)
                        els.append((nm, box))
            except Exception:
                pass
            walk(ch, d + 1)

    walk(root)


def enumerate_elements(win, cap=140):
    """Clickable elements of the window PLUS any open dropdown/context menu (those are separate
    top-level windows: classic menu class '#32768' or a MenuControl popup)."""
    els, seen = [], set()
    _collect(win, els, seen, cap)
    try:
        whandle = win.NativeWindowHandle
        for top in auto.GetRootControl().GetChildren():
            try:
                if top.IsOffscreen or top.NativeWindowHandle == whandle:
                    continue
                cn = (top.ClassName or "").lower()
                is_popup = (cn == "#32768" or top.ControlTypeName == "MenuControl"
                            or "popup" in cn or "flyout" in cn or "menu" in cn)
                if is_popup:
                    _collect(top, els, seen, cap)
            except Exception:
                pass
    except Exception:
        pass
    return els


def toks(s):
    s = " " + re.sub(r"[^a-z0-9+\-*/=%. ]", " ", s.lower()) + " "
    words = set(re.findall(r"(?<![a-z0-9])([a-z0-9]+)(?![a-z0-9])", s))
    syms = set(ch for ch in s if ch in SYMS)
    return words, syms, s


def el_keys(nm):
    fn = friendly(nm).lower()
    keys = set()
    for w in re.findall(r"[a-z]+|[0-9]+", fn):
        if w not in STOP and len(w) >= 1:
            keys.add(w)
    ksyms = set(ch for ch in fn if ch in SYMS)
    return keys, ksyms


def match_element(text, els):
    """Best fuzzy match of free text (model output OR raw instruction) to a UIA element. idx or -1."""
    words, syms, _ = toks(text)
    best, score = -1, 0
    for i, (nm, _) in enumerate(els):
        kw, ks = el_keys(nm)
        s = 0
        for k in kw:
            if k in words:
                s += len(k) + 2          # whole-token hit; longer = more specific
        for k in ks:
            if k in syms:
                s += 4
        if s > score:
            best, score = i, s
    return best if score >= 3 else -1


def ask_target(task):
    """Model names the target the instruction refers to (echoes it). Prefilled to avoid captioning."""
    payload = {
        "model": "moe-omni-500m",
        "messages": [
            {"role": "user", "content": f"What UI element does this instruction click? Instruction: {task}"},
            {"role": "assistant", "content": "The element is the"},
        ],
        "max_tokens": 8, "temperature": 0,
    }
    req = Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return (json.load(urlopen(req, timeout=180))["choices"][0]["message"]["content"] or "").strip()


def resolve(task, els):
    target = ask_target(task)
    idx, via = match_element(target, els), "model"
    if idx < 0:
        idx, via = match_element(task, els), "instruction"
    return idx, target, via


# ---------- typing + key presses ----------
EDIT_TYPES = {"EditControl", "DocumentControl"}
KEYMAP = {"enter": "enter", "return": "enter", "tab": "tab", "escape": "esc", "esc": "esc",
          "space": "space", "spacebar": "space", "backspace": "backspace", "delete": "delete",
          "del": "delete", "up": "up", "down": "down", "left": "left", "right": "right",
          "home": "home", "end": "end", "pageup": "pageup", "pagedown": "pagedown"}
TYPE_VERB = re.compile(r"^\s*(type|enter|write|input|fill(?:\s+in)?)\b", re.I)
KEY_VERB = re.compile(r"^\s*(?:press|hit|push)\s+(?:the\s+)?(" + "|".join(KEYMAP) + r")\b", re.I)


def extract_text(task):
    """The text to type: prefer quoted, else the words after the verb (minus a trailing 'into the X')."""
    m = re.search(r"['\"‘’“”]([^'\"‘’“”]+)", task)
    if m:
        return m.group(1)
    m = TYPE_VERB.match(task)
    rest = task[m.end():].strip() if m else task
    rest = re.split(r"\s+(?:in|into|on|to)\s+(?:the\s+)?\S", rest)[0]  # drop "into the search box"
    return rest.strip(" .:'\"")


def enumerate_edits(win):
    els, seen = [], set()

    def walk(c, d=0):
        if d > 26:
            return
        for ch in c.GetChildren():
            try:
                if ch.ControlTypeName in EDIT_TYPES and ch.IsEnabled and not ch.IsOffscreen:
                    r = ch.BoundingRectangle
                    box = (r.left, r.top, r.right, r.bottom)
                    if (r.right - r.left) > 8 and (r.bottom - r.top) > 8 and box not in seen:
                        seen.add(box)
                        els.append(((ch.Name or ch.AutomationId or "").strip(), box, ch))
            except Exception:
                pass
            walk(ch, d + 1)

    walk(win)
    return els


def find_field(win, task):
    """The edit field to type into: a named one if the task says 'into the X', else the largest."""
    edits = enumerate_edits(win)
    if not edits:
        return None
    m = re.search(r"\b(?:in|into|on|to)\s+(?:the\s+)?(.+)$", task, re.I)
    if m:
        named = m.group(1).strip()  # keep "search box"/"name field" intact for disambiguation
        idx = match_element(named, [(nm, bb) for nm, bb, _ in edits]) if named else -1
        if idx >= 0:
            return edits[idx]
    # else the biggest editable area (main text box / the sole input)
    return max(edits, key=lambda e: (e[1][2] - e[1][0]) * (e[1][3] - e[1][1]))


def do_type(win, task, execute):
    text = extract_text(task)
    field = find_field(win, task)
    if not field:
        print(f"   TYPE {text!r} -> no edit field found"); return
    nm, bb, ctrl = field
    cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
    print(f"   TYPE {text!r} -> field {nm or '(main)'!r} @ ({cx},{cy})")
    if execute:
        import pyautogui
        pyautogui.click(cx, cy); time.sleep(0.2)
        pyautogui.typewrite(text, interval=0.02)
        time.sleep(0.3)


def do_key(key, execute):
    print(f"   KEY  press {key!r}")
    if execute:
        import pyautogui
        pyautogui.press(key); time.sleep(0.3)


def main():
    args = sys.argv[1:]
    if "--list" in args:
        print("windows:\n  " + "\n  ".join(sorted(set(list_windows())))); return
    execute = "--execute" in args
    args = [a for a in args if a != "--execute"]
    win_name = args[0] if args else "--foreground"
    tasks = args[1:] or ["Click 7", "Click plus", "Click 9", "Click equals"]

    win = get_window(win_name)
    if not win:
        print(f"window {win_name!r} not found (try --list)"); return
    win.SetActive(); time.sleep(0.4)

    for n, task in enumerate(tasks, 1):
        print(f"[{n}] {task!r}")
        mk = KEY_VERB.match(task)
        if mk:                                       # press Enter/Tab/Esc/...
            do_key(KEYMAP[mk.group(1).lower()], execute); continue
        if TYPE_VERB.match(task):                    # type text into a field
            do_type(win, task, execute); continue
        els = enumerate_elements(win)                # click: re-enumerate (UI changes)
        idx, target, via = resolve(task, els)
        if idx < 0:
            print(f"   CLICK -> model:{target!r} -> NO MATCH among {len(els)} elements"); continue
        nm, bb = els[idx]
        cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
        print(f"   CLICK -> [{via}] {friendly(nm)!r} @ ({cx},{cy})")
        if execute:
            import pyautogui
            pyautogui.click(cx, cy); time.sleep(0.7)


if __name__ == "__main__":
    main()
