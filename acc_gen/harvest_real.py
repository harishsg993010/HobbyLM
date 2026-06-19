#!/usr/bin/env python3
"""Harvest REAL Windows UI-Automation trees from live apps, to use as grounded context for Gemini.

Instead of asking Gemini to imagine a realistic cluttered screen, we snapshot ACTUAL accessibility
trees (Settings pages, File Explorer, Calculator, Notepad, WordPad, Paint, Control Panel, a fresh
Chrome window) — chrome, decorative Text/Group/Image, disabled items, near-identical clusters and all
— exactly the raw distribution the inference harness sees. Each tree -> one line of real_trees.jsonl:
    {"label","window","screen","elements":[{"control_type","name","state"}...]}

Privacy: we LAUNCH fresh/neutral windows (system Settings, a system folder, Chrome on example.com) and
only READ their accessibility tree (no clicks). We do not snapshot your existing personal windows unless
you pass --include-open.

    python harvest_real.py                 # launch the safe set, snapshot, write real_trees.jsonl
    python harvest_real.py --keep-open     # don't close the windows afterward
"""
import argparse
import json
import os
import subprocess
import sys
import time

import uiautomation as auto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ui_harness import short_ctype, el_state  # reuse the same ctype/state logic the harness uses

# Raw capture keeps MORE than the harness filter (we WANT the chrome/context noise in training).
RAW_KEEP = {"Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "List", "ListItem",
            "MenuItem", "Menu", "Tab", "TabItem", "Slider", "Spinner", "Hyperlink", "Text",
            "TreeItem", "Document", "Image", "Group", "SplitButton", "ToggleButton",
            "ProgressBar", "Hyperlink", "TabItem"}

CHROME_EXE = None
for _p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
           r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
    if os.path.exists(_p):
        CHROME_EXE = _p
        break

# (label, launch callable or None, window-title substring, settle seconds)
def _settings(uri):
    return lambda: subprocess.Popen(["cmd", "/c", "start", "", uri])

def _explorer(path):
    return lambda: subprocess.Popen(["explorer.exe", path])

def _run(exe, *args):
    return lambda: subprocess.Popen([exe, *args])

TARGETS = [
    ("calculator",   _run("calc.exe"),                 "Calculator", 2.5),
    ("notepad",      _run("notepad.exe"),              "Notepad", 2.0),
    ("wordpad",      _run("write.exe"),                "WordPad", 3.0),
    ("paint",        _run("mspaint.exe"),              "Paint", 3.0),
    ("control_panel", _run("control.exe"),             "Control Panel", 3.0),
    ("settings_display",  _settings("ms-settings:display"),        "Settings", 3.5),
    ("settings_bluetooth", _settings("ms-settings:bluetooth"),     "Settings", 3.0),
    ("settings_network",  _settings("ms-settings:network-status"), "Settings", 3.0),
    ("settings_apps",     _settings("ms-settings:appsfeatures"),   "Settings", 3.0),
    ("settings_personalize", _settings("ms-settings:personalization"), "Settings", 3.0),
    ("settings_update",   _settings("ms-settings:windowsupdate"),  "Settings", 3.0),
    ("settings_sound",    _settings("ms-settings:sound"),          "Settings", 3.0),
    ("settings_power",    _settings("ms-settings:powersleep"),     "Settings", 3.0),
    ("settings_storage",  _settings("ms-settings:storagesense"),   "Settings", 3.0),
    ("settings_about",    _settings("ms-settings:about"),          "Settings", 3.0),
    ("settings_defaultapps", _settings("ms-settings:defaultapps"), "Settings", 3.0),
    ("settings_notifications", _settings("ms-settings:notifications"), "Settings", 3.0),
    ("settings_mouse",    _settings("ms-settings:mousetouchpad"),  "Settings", 3.0),
    ("settings_privacy",  _settings("ms-settings:privacy"),        "Settings", 3.0),
    ("explorer_windows",  _explorer(r"C:\Windows"),    "Windows", 3.0),
    ("explorer_system32", _explorer(r"C:\Windows\System32"), "System32", 3.0),
    ("explorer_programfiles", _explorer(r"C:\Program Files"), "Program Files", 3.0),
    ("explorer_thispc",   _explorer("shell:MyComputerFolder"), "This PC", 3.0),
    ("taskmgr",           _run("taskmgr.exe"),         "Task Manager", 3.5),
]
if CHROME_EXE:
    for _lab, _url, _t in [
        ("chrome_example", "https://example.com", 4.0),
        ("chrome_wikipedia", "https://en.wikipedia.org/wiki/Computer", 4.5),
        ("chrome_wikipedia_main", "https://www.wikipedia.org", 4.0),
        ("chrome_github", "https://github.com/about", 4.5),
        ("chrome_mdn", "https://developer.mozilla.org/en-US/", 4.5),
    ]:
        TARGETS.append((_lab, _run(CHROME_EXE, "--new-window", _url), "Chrome", _t))


def get_window(name):
    w = auto.WindowControl(searchDepth=1, SubName=name)
    if w.Exists(3):
        return w
    # Some apps (Settings/Explorer) are PaneControl or have the title deeper.
    for ctrl in (auto.PaneControl, auto.WindowControl):
        c = ctrl(searchDepth=1, SubName=name)
        if c.Exists(1):
            return c
    return None


def raw_snapshot(win, cap=100):
    els, seen, seen_nm = [], set(), set()

    def walk(c, d=0):
        if d > 34 or len(els) >= cap:
            return
        for ch in c.GetChildren():
            try:
                ctype = short_ctype(ch.ControlTypeName)
                nm = (ch.Name or ch.AutomationId or "").strip()
                r = ch.BoundingRectangle
                box = (r.left, r.top, r.right, r.bottom)
                vis = (r.right - r.left) > 2 and (r.bottom - r.top) > 2 and not ch.IsOffscreen
                nl = nm.lower()
                if (ctype in RAW_KEEP and nm and vis and box not in seen
                        and (ctype, nl) not in seen_nm):
                    seen.add(box); seen_nm.add((ctype, nl))
                    els.append({"control_type": ctype, "name": nm, "state": el_state(ch, ctype)})
            except Exception:
                pass
            walk(ch, d + 1)

    walk(win)
    return els


def serialize(window, els):
    lines = [f'[Window] "{window}"']
    for e in els:
        st = f"  ({e['state']})" if e.get("state") else ""
        lines.append(f'[{e["control_type"]}] "{e["name"]}"{st}')
    return "SCREEN:\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real_trees.jsonl")
    ap.add_argument("--keep-open", action="store_true", help="don't close windows after snapshot")
    ap.add_argument("--cap", type=int, default=100)
    args = ap.parse_args()

    out = []
    for label, launch, title, settle in TARGETS:
        proc = None
        try:
            if launch is not None:
                proc = launch()
            time.sleep(settle)
            win = get_window(title)
            if win is None:
                print(f"  [{label}] window {title!r} not found — skip", file=sys.stderr)
                continue
            try:
                win.SetTopmost(True); win.SetFocus()
            except Exception:
                pass
            time.sleep(0.6)
            els = raw_snapshot(win, cap=args.cap)
            if len(els) < 5:
                print(f"  [{label}] only {len(els)} elements — skip", file=sys.stderr)
            else:
                title_name = (win.Name or title).strip()
                out.append({"label": label, "window": title_name,
                            "screen": serialize(title_name, els), "elements": els})
                print(f"  [{label}] {len(els)} elements  ({title_name})", file=sys.stderr)
            if not args.keep_open:
                try:
                    win.GetWindowPattern().Close()  # reliable for start/explorer-hosted windows
                except Exception:
                    pass
        except Exception as e:
            print(f"  [{label}] error: {str(e)[:100]}", file=sys.stderr)
        finally:
            if proc is not None and not args.keep_open:
                try:
                    proc.terminate()
                except Exception:
                    pass

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nharvested {len(out)} real trees -> {args.out} "
          f"(avg {sum(len(r['elements']) for r in out)//max(len(out),1)} elements)", file=sys.stderr)


if __name__ == "__main__":
    main()
