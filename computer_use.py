"""Reliable computer-use harness for the 500M — built from the tool-call eval findings.

What the Nemotron eval taught us:
  * the model gets the FUNCTION NAME right most of the time, but ARGUMENT VALUES drift (type mismatch
    "1234" vs 1234, hallucinated values languagecode "en" vs "es", extra/missing optional params).
  * multi-shot prompting HURTS (single-shot is the only trained format).
  * a deterministic fuzzy router selects the right tool 10/10 where the model managed 3/10.

So this harness:
  1. SELECTS the tool deterministically (fuzzy trigger + URL regex), with the model only as a tiebreaker
     when no trigger fires (rare).
  2. EXTRACTS arguments deterministically from the instruction (NOT from the model — that's the weak part).
  3. EXECUTES a safe subset of real Windows actions (`--execute`); dry-run prints the resolved command.

Run:
  python computer_use.py                 # dry-run: show tool + args + the OS command for each demo
  python computer_use.py --execute       # actually perform the SAFE ops (web/search/screenshot/settings)
  python computer_use.py "open notepad"  # one instruction
"""
import sys, os, re, time, json, webbrowser, subprocess
from urllib.parse import quote

# ---- 1. DETERMINISTIC TOOL SELECTION (the workhorse; 10/10 in test_computer_tools.py) ---------------
TRIGGERS = {
    "open_app":         [("open", 2), ("launch", 3), ("start", 2), ("run", 2)],
    "close_app":        [("close", 3), ("quit", 3), ("exit", 3), ("kill", 3)],
    "open_website":     [("go to", 3), ("visit", 3), ("website", 2), ("browse", 2), ("open url", 3)],
    "web_search":       [("search", 3), ("look up", 3), ("find online", 3), ("web for", 3), ("google for", 3)],
    "take_screenshot":  [("screenshot", 4), ("capture", 3), ("screen shot", 4), ("snip", 3), ("grab the screen", 4)],
    "set_volume":       [("volume", 4), ("loud", 3), ("mute", 3)],
    "lock_screen":      [("lock", 4)],
    "open_settings":    [("settings", 3), ("preferences", 3), ("configure", 2)],
    "create_folder":    [("folder", 4), ("directory", 3), ("mkdir", 4)],
    "play_pause_media": [("pause", 4), ("resume", 3), ("play", 2)],
}
URL_RE = re.compile(r"\b([\w-]+\.(?:com|org|net|io|gov|edu|co|dev|ai))\b|https?://\S+", re.I)


def route(instr):
    ql = instr.lower()
    if URL_RE.search(ql) and not any(t in ql for t in ("search", "look up", "google for")):
        return "open_website"
    words = re.findall(r"[a-z]+", ql)
    best, bs = None, 0.0
    for name, trigs in TRIGGERS.items():
        s = 0.0
        for phrase, w in trigs:
            if phrase in ql:
                s += w + 0.5 * phrase.count(" ")
            elif " " not in phrase and words:
                from difflib import SequenceMatcher
                if max(SequenceMatcher(None, phrase, x).ratio() for x in words) >= 0.86:
                    s += w * 0.6
        if s > bs:
            bs, best = s, name
    return best


# ---- 2. DETERMINISTIC ARGUMENT EXTRACTION (the model drifts here, so we don't ask it) ----------------
_STRIP = r"^(please\s+|can you\s+|could you\s+)?(open|launch|start|run|close|quit|exit|kill|go to|" \
         r"visit|browse to|search( the web)?( for)?|look up|google( for)?|find( online)?|take|capture|" \
         r"grab|set|turn|make( a)?( new)?|create( a)?( new)?|show)\b"
SETTINGS_PAGES = {
    "display": "display", "screen": "display", "sound": "sound", "audio": "sound",
    "wifi": "network-wifi", "wi-fi": "network-wifi", "network": "network",
    "bluetooth": "bluetooth", "battery": "batterysaver", "power": "powersleep",
    "update": "windowsupdate", "notification": "notifications", "app": "appsfeatures",
    "privacy": "privacy", "storage": "storagesense", "theme": "themes", "background": "personalization-background",
}
KNOWN_APPS = {  # name fragment -> launch target (os.startfile / start)
    "calculator": "calc", "calc": "calc", "notepad": "notepad", "paint": "mspaint",
    "explorer": "explorer", "file explorer": "explorer", "cmd": "cmd", "terminal": "wt",
    "task manager": "taskmgr", "chrome": "chrome", "edge": "msedge", "word": "winword",
    "excel": "excel", "spotify": "spotify", "settings": "ms-settings:",
}


def extract_args(tool, instr):
    q = instr.strip()
    if tool in ("take_screenshot", "lock_screen", "play_pause_media"):
        return {}
    if tool == "set_volume":
        m = re.search(r"(\d{1,3})", q)
        lvl = max(0, min(100, int(m.group(1)))) if m else (0 if "mute" in q.lower() else 50)
        return {"level": lvl}
    if tool == "open_website":
        m = URL_RE.search(q)
        url = m.group(0) if m else q
        if not url.lower().startswith("http"):
            url = "https://" + url
        return {"url": url}
    if tool == "web_search":
        query = re.sub(_STRIP, "", q, flags=re.I).strip(" .?!")
        query = re.sub(r"^(the web\s+)?(for\s+)?", "", query, flags=re.I).strip()
        return {"query": query or q}
    if tool == "open_settings":
        ql = q.lower()
        page = next((v for k, v in SETTINGS_PAGES.items() if k in ql), "")
        return {"page": page}
    # open_app / close_app / create_folder -> the remaining noun phrase
    rest = re.sub(_STRIP, "", q, flags=re.I)
    rest = re.sub(r"\b(the|app|application|program|window|called|named|a new|new)\b", " ", rest, flags=re.I)
    if tool == "create_folder":                       # also drop the trigger noun itself
        rest = re.sub(r"\b(folder|directory)\b", " ", rest, flags=re.I)
    rest = re.sub(r"\s+", " ", rest).strip(" .?!\"'")
    if tool == "create_folder":
        return {"name": rest or "New Folder"}
    return {"name": rest}


# ---- 3. SAFE EXECUTION (only the non-destructive ops actually run) -----------------------------------
SAFE = {"open_website", "web_search", "take_screenshot", "open_settings", "open_app"}


def execute(tool, args, do_real):
    """Return (os_command_description, status). Only SAFE ops run when do_real; others are dry-run."""
    if tool == "open_website":
        cmd = f"browser -> {args['url']}"
        if do_real:
            webbrowser.open_new_tab(args["url"])
        return cmd, "opened" if do_real else "dry-run"
    if tool == "web_search":
        url = "https://www.google.com/search?q=" + quote(args["query"])
        if do_real:
            webbrowser.open_new_tab(url)
        return f"browser -> {url}", "searched" if do_real else "dry-run"
    if tool == "take_screenshot":
        path = os.path.join(os.path.expanduser("~"), "Desktop", f"shot_{int(time.time())}.png")
        if do_real:
            try:
                from PIL import ImageGrab
                ImageGrab.grab().save(path)
            except Exception as e:
                return f"screenshot -> {path}", f"FAILED ({type(e).__name__})"
        return f"screenshot -> {path}", "saved" if do_real else "dry-run"
    if tool == "open_settings":
        uri = "ms-settings:" + (args.get("page") or "")
        if do_real:
            os.startfile(uri)
        return f"shell -> {uri}", "opened" if do_real else "dry-run"
    if tool == "open_app":
        name = args.get("name", "")
        nl = name.lower()
        if nl.startswith("http") or URL_RE.search(nl):              # url -> browser
            url = name if nl.startswith("http") else "https://" + name
            if do_real:
                webbrowser.open_new_tab(url)
            return f"browser -> {url}", "opened" if do_real else "dry-run"
        if "setting" in nl:                                         # settings -> ms-settings:
            page = next((v for k, v in SETTINGS_PAGES.items() if k in nl), "")
            uri = "ms-settings:" + page
            if do_real:
                os.startfile(uri)
            return f"shell -> {uri}", "opened" if do_real else "dry-run"
        target = KNOWN_APPS.get(nl, name)                           # else launch the app
        if do_real:
            try:
                subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
            except Exception as e:
                return f"start {target}", f"FAILED ({type(e).__name__})"
        return f"start {target}", "launched" if do_real else "dry-run"
    # close_app / set_volume / lock_screen / create_folder / play_pause -> resolved but NOT auto-run
    if tool == "close_app":
        return f"taskkill /IM {args['name']}.exe", "dry-run (guarded)"
    if tool == "set_volume":
        return f"set system volume = {args['level']}%", "dry-run (needs pycaw)"
    if tool == "lock_screen":
        return "user32.LockWorkStation()", "dry-run (guarded)"
    if tool == "create_folder":
        path = os.path.join(os.path.expanduser("~"), "Desktop", args["name"])
        return f"mkdir {path}", "dry-run (guarded)"
    if tool == "play_pause_media":
        return "media play/pause key", "dry-run (guarded)"
    return "(no handler)", "skip"


# ---- THE MODEL DRIVES (single-shot intent call — the regime that worked: 5/8 zero-shot) -------------
from urllib.request import Request, urlopen
from difflib import get_close_matches
API = "http://127.0.0.1:11250/v1/chat/completions"
NAMES = list(TRIGGERS.keys())


def _t(name, desc, props=None, req=None):
    p = {"type": "object", "properties": props or {}}
    if req:
        p["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}


_S, _I = {"type": "string"}, {"type": "integer"}
# DISTINCT, NON-OVERLAPPING intents (mobile-style — the regime the model drives well). Crucially only
# ONE "open" tool: the generic set's open_app/open_website/open_settings/web_search all matched "open"
# and the model collapsed every query into open_website. open_app now handles apps AND urls; web_search
# is the only search tool. This is what "model in the loop" needs to actually work.
TOOLSPEC = [
    _t("open_app", "Launches an app, website, or settings page by name (e.g. 'chrome', 'youtube.com', 'wifi settings').", {"name": _S}, ["name"]),
    _t("web_search", "Searches the web for information.", {"query": _S}, ["query"]),
    _t("take_screenshot", "Takes a screenshot of the screen."),
    _t("set_volume", "Sets the system volume from 0 to 100.", {"level": _I}, ["level"]),
    _t("lock_screen", "Locks the computer screen."),
    _t("create_folder", "Creates a new folder.", {"name": _S}, ["name"]),
    _t("play_pause_media", "Plays or pauses media playback."),
]
PARAMS = {t["function"]["name"]: list(t["function"]["parameters"]["properties"].keys()) for t in TOOLSPEC}


def model_call(instr):
    """Ask the MODEL to pick the tool + args (single-shot, the trained format). -> (name|None, args)."""
    body = json.dumps({"model": "moe-omni-500m", "messages": [{"role": "user", "content": instr}],
                       "tools": TOOLSPEC, "tool_choice": "required", "max_tokens": 60, "temperature": 0}).encode()
    try:
        m = json.load(urlopen(Request(API, data=body, headers={"Content-Type": "application/json"}),
                              timeout=60))["choices"][0]["message"]
    except Exception as e:
        return None, {}, f"api-err:{type(e).__name__}"
    tcs = m.get("tool_calls")
    if not tcs:
        return None, {}, "no-call"
    f = tcs[0]["function"]
    try:
        a = json.loads(f.get("arguments") or "{}")
    except Exception:
        a = {}
    return f["name"], (a if isinstance(a, dict) else {}), "model"


def repair(name, margs, instr):
    """The model DECIDES the action (selection is its strength). Args come from the INSTRUCTION
    deterministically (the eval proved the model drifts on arg values); model args only fill gaps
    the extractor left. Abstention/garbled name -> deterministic route() fallback."""
    if name in NAMES:
        tool, fixed = name, "model-picked"
    else:
        cand = get_close_matches((name or "").lower(), NAMES, 1, 0.6)
        tool = cand[0] if cand else route(instr)
        fixed = f"recover:{name}->{tool}"
    if not tool:
        return None, {}, fixed
    out = dict(extract_args(tool, instr))      # DETERMINISTIC args are primary
    for k, v in (margs or {}).items():         # model args only fill what the extractor missed
        if k in PARAMS.get(tool, []) and (k not in out or out[k] in (None, "", [])) and v not in (None, "", []):
            out[k] = v
    if tool == "set_volume":
        try:
            out["level"] = max(0, min(100, int(out.get("level", 50))))
        except Exception:
            out["level"] = 50
    return tool, out, fixed


def run(instr, do_real, use_model=True):
    if use_model:
        mname, margs, src = model_call(instr)
        tool, args, fixed = repair(mname, margs, instr)
        note = f"[model:{mname}({json.dumps(margs)}) {src}; {fixed}]"
    else:
        tool, args, note = route(instr), None, "[deterministic]"
        if tool:
            args = extract_args(tool, instr)
    if not tool:
        print(f"  {instr!r:40s} -> (no tool)  {note}")
        return
    cmd, status = execute(tool, args, do_real)
    print(f"  {instr!r:40s} -> {tool}({json.dumps(args)})")
    print(f"  {'':40s}    {status:18s} {cmd}")
    print(f"  {'':40s}    {note}")


DEMO = [
    "open chrome", "open notepad", "open youtube.com", "search the web for pasta recipes",
    "take a screenshot", "set the volume to 30", "lock my screen", "open wifi settings",
    "make a new folder called Projects", "pause the music",
]

if __name__ == "__main__":
    do_real = "--execute" in sys.argv
    use_model = "--no-model" not in sys.argv      # model DRIVES by default; deterministic only repairs
    items = [a for a in sys.argv[1:] if not a.startswith("--")] or DEMO
    print(f"mode: {'EXECUTE (safe ops)' if do_real else 'DRY-RUN'} | "
          f"{'MODEL-DRIVEN (+ deterministic repair)' if use_model else 'deterministic-only'}\n")
    for it in items:
        run(it, do_real, use_model)
