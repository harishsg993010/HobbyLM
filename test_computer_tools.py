"""Computer-use intent tools + fuzzy resolution + zero-shot vs multi-shot prompting.

The model generalizes function-calling to high-level INTENT functions within its training domain. Here:
 1. A full computer-use intent toolset (open_app/close_app/website/search/screenshot/volume/lock/...).
 2. FUZZY resolution: map the model's (possibly typo'd) function name to the closest real tool.
 3. A/B: ZERO-shot (just tools+query) vs MULTI-shot (a few in-context query->call examples first) to
    see if few-shot fixes the dominant-attractor mis-routing.
"""
import sys, re, json, difflib
from urllib.request import Request, urlopen

API = "http://127.0.0.1:11250/v1/chat/completions"


def fn(name, desc, props=None, req=None):
    p = {"type": "object", "properties": props or {}}
    if req:
        p["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}


S = {"type": "string"}
I = {"type": "integer"}
TOOLS = [
    fn("open_app", "Opens an application by name.", {"name": S}, ["name"]),
    fn("close_app", "Closes an application by name.", {"name": S}, ["name"]),
    fn("open_website", "Opens a website URL in the browser.", {"url": S}, ["url"]),
    fn("web_search", "Searches the web for a query.", {"query": S}, ["query"]),
    fn("take_screenshot", "Takes a screenshot of the screen."),
    fn("set_volume", "Sets the system volume (0-100).", {"level": I}, ["level"]),
    fn("lock_screen", "Locks the computer screen."),
    fn("open_settings", "Opens a Windows settings page.", {"page": S}, ["page"]),
    fn("create_folder", "Creates a new folder.", {"name": S}, ["name"]),
    fn("play_pause_media", "Toggles play/pause of media."),
]
NAMES = [t["function"]["name"] for t in TOOLS]

# in-context examples for multi-shot (diverse tools -> teach the mapping, fight the attractor)
FEWSHOT = [
    ("Launch Spotify", "open_app", {"name": "Spotify"}),
    ("Find me a pizza recipe online", "web_search", {"query": "pizza recipe"}),
    ("Turn the volume down to 10", "set_volume", {"level": 10}),
    ("Capture my screen", "take_screenshot", {}),
]

CASES = [
    ("Open Google Chrome", "open_app"),
    ("Close the calculator", "close_app"),
    ("Go to youtube.com", "open_website"),
    ("Search the web for pasta recipes", "web_search"),
    ("Take a screenshot", "take_screenshot"),
    ("Set the volume to 30", "set_volume"),
    ("Lock my screen", "lock_screen"),
    ("Open the display settings", "open_settings"),
    ("Make a new folder called Projects", "create_folder"),
    ("Pause the music", "play_pause_media"),
]


def fuzzy_name(name):
    if name in NAMES:
        return name
    m = difflib.get_close_matches((name or "").strip().lower(), NAMES, n=1, cutoff=0.55)
    return m[0] if m else name


# ---- DETERMINISTIC fuzzy intent router (the real workhorse; model-free) -------------------
# Each tool: weighted trigger phrases. Multi-word phrase present in the instruction = strong hit;
# otherwise fuzzy per-word match (difflib) catches typos/inflections. A URL/domain forces website.
TRIGGERS = {
    "open_app":         [("open", 2), ("launch", 2), ("start", 1), ("app", 1)],
    "close_app":        [("close", 2), ("quit", 2), ("exit", 2), ("kill", 2)],
    "open_website":     [("go to", 2), ("visit", 2), ("website", 2), ("browse", 2), ("open url", 2)],
    "web_search":       [("search", 3), ("look up", 2), ("find online", 2), ("web for", 2)],
    "take_screenshot":  [("screenshot", 3), ("capture", 2), ("screen shot", 3), ("snip", 2)],
    "set_volume":       [("volume", 3), ("loud", 2), ("sound", 1), ("mute", 2)],
    "lock_screen":      [("lock", 3)],
    "open_settings":    [("settings", 3), ("preferences", 2), ("configure", 2)],
    "create_folder":    [("folder", 3), ("directory", 2), ("mkdir", 3)],
    "play_pause_media": [("pause", 3), ("play", 2), ("resume", 2), ("music", 1), ("media", 1)],
}
URL_RE = re.compile(r"\b\w[\w-]*\.(com|org|net|io|gov|edu|co)\b|https?://", re.I)


def fuzzy_route(q):
    ql = q.lower()
    if URL_RE.search(ql):
        return "open_website"
    words = re.findall(r"[a-z]+", ql)
    best, bs = None, 0.0
    for name, trigs in TRIGGERS.items():
        s = 0.0
        for phrase, w in trigs:
            if phrase in ql:
                s += w + 0.5 * phrase.count(" ")          # phrase present (favor multi-word)
            elif " " not in phrase:
                r = max((difflib.SequenceMatcher(None, phrase, x).ratio() for x in words), default=0)
                if r >= 0.84:
                    s += w * 0.6                            # fuzzy single-word (typo/inflection)
        if s > bs:
            bs, best = s, name
    return best


def call(query, multishot):
    msgs = []
    if multishot:
        for u, nm, a in FEWSHOT:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": "ex", "type": "function",
                                         "function": {"name": nm, "arguments": json.dumps(a)}}]})
    msgs.append({"role": "user", "content": query})
    body = json.dumps({"model": "moe-omni-500m", "messages": msgs, "tools": TOOLS,
                       "tool_choice": "required", "max_tokens": 40, "temperature": 0}).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    m = json.load(urlopen(req, timeout=180))["choices"][0]["message"]
    tcs = m.get("tool_calls")
    if not tcs:
        return None, {}
    f = tcs[0]["function"]
    try:
        a = json.loads(f.get("arguments") or "{}")
    except Exception:
        a = {}
    return fuzzy_name(f["name"]), a


def fuzzy_only():
    """Deterministic router only — no server needed."""
    f = 0
    print(f"{'query':34s} | {'expected':16s} | {'FUZZY (model-free)':18s}")
    print("-" * 76)
    for q, exp in CASES:
        fn_ = fuzzy_route(q)
        ok = fn_ == exp
        f += ok
        print(f"{q:34s} | {exp:16s} | {str(fn_)[:16]:16s}{'OK' if ok else 'x':>2}")
    print("-" * 76)
    print(f"FUZZY (deterministic): {f}/{len(CASES)}")


def main():
    z = ms = f = 0
    print(f"{'query':32s} | {'expected':16s} | {'ZERO':12s} | {'MULTI':12s} | {'FUZZY':12s}")
    print("-" * 100)
    for q, exp in CASES:
        zn, _ = call(q, False)
        mn, _ = call(q, True)
        fn_ = fuzzy_route(q)
        z += zn == exp
        ms += mn == exp
        f += fn_ == exp
        mark = lambda r: "OK" if r == exp else "x"
        print(f"{q:32s} | {exp:16s} | {str(zn)[:10]:10s}{mark(zn):>2} | "
              f"{str(mn)[:10]:10s}{mark(mn):>2} | {str(fn_)[:10]:10s}{mark(fn_):>2}")
    n = len(CASES)
    print("-" * 100)
    print(f"ZERO-SHOT: {z}/{n}    MULTI-SHOT: {ms}/{n}    FUZZY: {f}/{n}")


if __name__ == "__main__":
    if "--fuzzy-only" in sys.argv:
        fuzzy_only()
    else:
        main()
