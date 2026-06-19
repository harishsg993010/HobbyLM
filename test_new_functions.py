"""Does the model generalize function-calling to NEW functions it was NOT fine-tuned on?

It was trained on 7 mobile actions (flashlight/wifi/map/contact/email/calendar). Here we offer a
DIFFERENT set of same-style mobile-assistant functions (alarm/timer/music/weather/call/app/volume/
screenshot) and see whether it (a) calls the right NEW function, (b) maps onto a trained one, or
(c) emits garbage. Single-turn, structured tool-calling over the API.
"""
import json
from urllib.request import Request, urlopen

API = "http://127.0.0.1:11250/v1/chat/completions"

def fn(name, desc, props=None, req=None):
    p = {"type": "object", "properties": props or {}}
    if req:
        p["required"] = req
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}

TOOLS = [
    fn("set_alarm", "Sets an alarm for a given time.", {"time": {"type": "string"}}, ["time"]),
    fn("set_timer", "Starts a countdown timer.", {"duration": {"type": "string"}}, ["duration"]),
    fn("play_music", "Plays music matching a query.", {"query": {"type": "string"}}, ["query"]),
    fn("get_weather", "Gets the current weather for a location.", {"location": {"type": "string"}}, ["location"]),
    fn("make_phone_call", "Calls a contact by name.", {"name": {"type": "string"}}, ["name"]),
    fn("open_app", "Opens an application by name.", {"app_name": {"type": "string"}}, ["app_name"]),
    fn("take_screenshot", "Takes a screenshot of the screen."),
    fn("set_volume", "Sets the system volume level (0-100).", {"level": {"type": "integer"}}, ["level"]),
]
NEW_NAMES = {t["function"]["name"] for t in TOOLS}

CASES = [
    ("Set an alarm for 7 in the morning", "set_alarm"),
    ("Set a timer for 10 minutes", "set_timer"),
    ("Play some jazz music", "play_music"),
    ("What is the weather in Tokyo", "get_weather"),
    ("Call mom", "make_phone_call"),
    ("Open the camera app", "open_app"),
    ("Take a screenshot", "take_screenshot"),
    ("Turn the volume up to 50", "set_volume"),
]


def call(task):
    body = json.dumps({"model": "moe-omni-500m", "messages": [{"role": "user", "content": task}],
                       "tools": TOOLS, "tool_choice": "required", "max_tokens": 40, "temperature": 0}).encode()
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
    return f["name"], a


def main():
    right = newcall = 0
    for task, exp in CASES:
        name, args = call(task)
        is_new = name in NEW_NAMES
        ok = (name == exp)
        right += ok
        newcall += is_new
        tag = "OK" if ok else ("new-but-wrong" if is_new else "->TRAINED/garbage")
        print(f"{task:36s} exp={exp:16s} got={str(name):20s} {tag}")
        print(f"{'':36s} args={args}")
    print("-" * 84)
    print(f"correct: {right}/{len(CASES)}   called a NEW (untrained) function: {newcall}/{len(CASES)}")


if __name__ == "__main__":
    main()
