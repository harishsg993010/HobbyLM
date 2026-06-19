"""UI-grounding agentic harness for the moe-omni-500m vision API.

The model has UI grounding (trained on Aria-UI): given a screenshot + an instruction, it outputs a click
point `(cx, cy)` normalized to 0-1000 (fraction of image * 1000). That makes the "tool" the simplest
possible for the agent — it ONLY has to say WHERE to click; this harness handles the screenshot and the
click. Loop = screenshot -> ask model -> click -> (UI changes) -> screenshot -> ...

Dry-run by default: it annotates the predicted point on the screenshot (ui_agent_step*.png) so you can
SEE if the grounding is right. Pass --execute to actually move the mouse + click (3s countdown; move the
mouse to a screen corner to abort via pyautogui's failsafe).

Usage:
    python ui_agent.py "Click the New chat button" "Click the message box"
    python ui_agent.py --execute "Click the Load button"
"""
import sys, re, io, json, time, base64, ctypes
from ctypes import wintypes
from urllib.request import Request, urlopen
from PIL import ImageGrab, ImageDraw

API = "http://127.0.0.1:11250/v1/chat/completions"

ctypes.windll.user32.SetProcessDPIAware()
SW = ctypes.windll.user32.GetSystemMetrics(0)
SH = ctypes.windll.user32.GetSystemMetrics(1)

# Capture region: full screen by default, or a window matched by --window <title-substr>.
# The model was trained on focused app screenshots, so a single window grounds far better than the
# whole busy desktop. Coords (0-1000) are mapped back into this region.
REGION = (0, 0, SW, SH)


user32 = ctypes.windll.user32


def find_window(title_substr):
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_substr.lower() in buf.value.lower():
                    r = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(r))
                    if r.right - r.left > 100 and r.bottom - r.top > 100:
                        found.append((hwnd, buf.value, (r.left, r.top, r.right, r.bottom)))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def minimize_matching(substrs):
    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if any(s.lower() in buf.value.lower() for s in substrs):
                    user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
        return True
    user32.EnumWindows(cb, 0)


def focus_window(hwnd):
    user32.ShowWindow(hwnd, 6); time.sleep(0.15)
    user32.ShowWindow(hwnd, 9); time.sleep(0.15)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    time.sleep(0.5)


def screenshot():
    return ImageGrab.grab(bbox=REGION).convert("RGB")


def ask_click(img, instruction):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": "moe-omni-500m",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]},
            # PREFILL "(" — forces the trained grounding format and beats the captioning prior
            {"role": "assistant", "content": "("},
        ],
        "max_tokens": 12,
        "temperature": 0,
    }
    req = Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    cont = json.load(urlopen(req, timeout=180))["choices"][0]["message"]["content"] or ""
    return "(" + cont  # prepend the prefill so the (cx, cy) parse sees the full point


def parse_point(txt):
    m = re.search(r"\(?\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)?", txt or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def annotate(img, px, py, path):
    d = ImageDraw.Draw(img)
    r = 26
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=5)
    d.line([px - r - 14, py, px + r + 14, py], fill=(255, 0, 0), width=3)
    d.line([px, py - r - 14, px, py + r + 14], fill=(255, 0, 0), width=3)
    img.save(path)


def step(instruction, execute, idx):
    print(f"\n[step {idx}] {instruction!r}")
    img = screenshot()
    raw = ask_click(img, instruction)
    print(f"  model -> {raw.strip()!r}")
    pt = parse_point(raw)
    if not pt:
        print("  (no click point parsed)")
        return
    cx, cy = pt
    rx, ry = REGION[0], REGION[1]
    rw, rh = REGION[2] - REGION[0], REGION[3] - REGION[1]
    mx, my = int(cx / 1000 * rw), int(cy / 1000 * rh)  # marker on the captured image
    px, py = rx + mx, ry + my                          # screen-absolute click point
    out = f"ui_agent_step{idx}.png"
    annotate(img.copy(), mx, my, out)
    print(f"  ({cx},{cy})/1000  ->  screen ({px},{py})   annotated: {out}")
    if execute:
        import pyautogui
        pyautogui.FAILSAFE = True
        print("  clicking in 3s (move mouse to a corner to abort)...")
        time.sleep(3)
        pyautogui.click(px, py)
        print("  clicked.")
        time.sleep(1.0)  # let the UI react before the next screenshot


if __name__ == "__main__":
    args = sys.argv[1:]
    execute = "--execute" in args
    win = None
    if "--window" in args:
        wi = args.index("--window")
        win = args[wi + 1]
        args = args[:wi] + args[wi + 2:]
    if win:
        # clear common overlap-causers, then bring the target to the front for a clean capture
        minimize_matching(["Claude", "Visual Studio", "Cursor", " Code", "Terminal", "PowerShell", "Edge", "Chrome"])
        time.sleep(0.5)
        found = find_window(win)
        if found:
            hwnd, title, _ = found
            focus_window(hwnd)
            r = wintypes.RECT(); user32.GetWindowRect(hwnd, ctypes.byref(r))
            REGION = (r.left, r.top, r.right, r.bottom)
            print(f"window: {title!r}  region {REGION}")
        else:
            print(f"window {win!r} not found — using full screen")
    instrs = [a for a in args if not a.startswith("--")]
    if not instrs:
        instrs = ["Click the New chat button", "Click the message input box at the bottom"]
    rw, rh = REGION[2] - REGION[0], REGION[3] - REGION[1]
    print(f"capture {rw}x{rh} | mode: {'EXECUTE (will click)' if execute else 'dry-run (annotate only)'}")
    for i, ins in enumerate(instrs, 1):
        step(ins, execute, i)
