"""Rigorous UI-grounding test on a CLEAN in-distribution app (Windows Calculator).

Tests whether the model's Aria-UI grounding actually varies CORRECTLY per element (vs a center prior):
asks for several distinctly-located buttons and annotates all predicted points on ONE screenshot.
Also tests a SIMPLE function-calling `click(x,y)` tool vs the direct `(cx,cy)` grounding output.

Run:  python ui_test.py
"""
import re, io, json, time, base64, ctypes, subprocess
from ctypes import wintypes
from urllib.request import Request, urlopen
from PIL import ImageGrab, ImageDraw

API = "http://127.0.0.1:11250/v1/chat/completions"
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

CLICK_TOOL = [{
    "type": "function",
    "function": {
        "name": "click",
        "description": "Click a point on the screen, given as normalized 0-1000 coordinates.",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    },
}]


def minimize_matching(substrs):
    """Minimize any visible top-level window whose title contains one of substrs (clears overlaps)."""
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


def b64_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def grounding(img, instruction, use_tool):
    """Return (cx,cy) in 0-1000, plus the raw text/struct, via direct grounding or a click() tool."""
    msg = {"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64_jpeg(img)}},
    ]}
    payload = {"model": "moe-omni-500m", "messages": [msg], "max_tokens": 40, "temperature": 0}
    if use_tool:
        payload["tools"] = CLICK_TOOL
        payload["tool_choice"] = "required"
    req = Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    m = json.load(urlopen(req, timeout=180))["choices"][0]["message"]
    if use_tool and m.get("tool_calls"):
        try:
            a = json.loads(m["tool_calls"][0]["function"]["arguments"] or "{}")
            return (int(a["x"]), int(a["y"])), json.dumps(a)
        except Exception:
            return None, str(m.get("tool_calls"))
    txt = m.get("content") or ""
    mt = re.search(r"\(?\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)?", txt)
    return (int(mt.group(1)), int(mt.group(2))) if mt else None, txt.strip()


COLORS = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 160, 0), (200, 0, 200), (0, 200, 200)]


def main():
    # clear overlapping windows (the Claude desktop app + the hobby-chat window keep covering the screen;
    # the API server keeps running even when the hobby-chat window is minimized)
    minimize_matching(["Claude", "hobby-chat", "Visual Studio", "Code", "Terminal", "PowerShell"])
    time.sleep(0.6)
    # launch a clean, single, in-distribution app
    subprocess.Popen("calc.exe", shell=True)
    time.sleep(2.5)
    w = find_window("Calculator")
    if not w:
        print("Calculator window not found"); return
    hwnd, title, region = w
    # force calc to the front: minimize -> restore tends to beat the foreground lock
    user32.ShowWindow(hwnd, 6); time.sleep(0.2)
    user32.ShowWindow(hwnd, 9); time.sleep(0.2)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    time.sleep(1.0)
    r = wintypes.RECT(); user32.GetWindowRect(hwnd, ctypes.byref(r))
    region = (r.left, r.top, r.right, r.bottom)
    rw, rh = region[2] - region[0], region[3] - region[1]
    print(f"window {title!r}  region {region}  ({rw}x{rh})")

    img = ImageGrab.grab(bbox=region).convert("RGB")
    img.save("ui_test_clean.png")

    targets = ["Click the number 7", "Click the number 0", "Click the plus button",
               "Click the equals button", "Click the number 9"]

    for use_tool in (False, True):
        mode = "function-call click()" if use_tool else "direct (cx,cy)"
        print(f"\n===== mode: {mode} =====")
        canvas = img.copy()
        d = ImageDraw.Draw(canvas)
        for i, instr in enumerate(targets):
            pt, raw = grounding(img, instr, use_tool)
            if not pt:
                print(f"  {instr:24s} -> NO POINT  (raw={raw!r})"); continue
            cx, cy = pt
            mx, my = int(cx / 1000 * rw), int(cy / 1000 * rh)
            col = COLORS[i % len(COLORS)]
            d.ellipse([mx - 18, my - 18, mx + 18, my + 18], outline=col, width=4)
            d.text((mx + 20, my - 8), instr.replace("the ", "").replace(" button", ""), fill=col)
            print(f"  {instr:24s} -> ({cx:4d},{cy:4d})/1000 -> px ({mx:4d},{my:4d})   raw={raw!r}")
        out = f"ui_test_{'tool' if use_tool else 'direct'}.png"
        canvas.save(out)
        print(f"  annotated -> {out}")


if __name__ == "__main__":
    main()
