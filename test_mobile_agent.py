"""Mobile-actions agent over the hobby-chat OpenAI API.

Exposes the EXACT 7 tools the model was fine-tuned on (google/mobile-actions) — the regime where the
500M is reliable. Runs the full loop: natural request -> model emits a tool_call -> execute -> the
model summarizes. Mock execution by default; `--execute` actually performs the safe ones (open Wi-Fi
settings, open the map in a browser).
"""
import sys, os, json, re, tempfile, webbrowser, subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen

API = "http://127.0.0.1:11250/v1/chat/completions"


def _ics_dt(s):
    """ISO-ish '2026-06-15T14:00:00' -> '20260615T140000' (calendar .ics)."""
    digits = re.sub(r"[^0-9T]", "", (s or "").replace(" ", "T"))
    d, _, t = digits.partition("T")
    d = (d + "00000000")[:8]
    t = (t + "000000")[:6]
    return d + "T" + t


def _open(path_or_url):
    try:
        os.startfile(path_or_url)  # opens the default handler (mail/calendar/contacts/browser)
    except Exception:
        webbrowser.open(path_or_url)


def _tmp(suffix, content):
    fd, p = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return p

# schemas copied 1:1 from the training data (google/mobile-actions)
TOOLS = [
    {"type": "function", "function": {"name": "turn_on_flashlight",
        "description": "Turns the flashlight on.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "turn_off_flashlight",
        "description": "Turns the flashlight off.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "open_wifi_settings",
        "description": "Opens the Wi-Fi settings.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "show_map",
        "description": "Shows a location on the map.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "create_contact",
        "description": "Creates a contact in the phone's contact list.",
        "parameters": {"type": "object", "properties": {
            "first_name": {"type": "string"}, "last_name": {"type": "string"},
            "phone_number": {"type": "string"}, "email": {"type": "string"}},
            "required": ["first_name", "last_name"]}}},
    {"type": "function", "function": {"name": "send_email",
        "description": "Sends an email.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject"]}}},
    {"type": "function", "function": {"name": "create_calendar_event",
        "description": "Creates a new calendar event.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "datetime": {"type": "string"}},
            "required": ["title", "datetime"]}}},
]

DO_REAL = "--execute" in sys.argv


def execute(name, args):
    """Perform the action as a REAL Windows op (with --execute), return a JSON result.

    The 7 mobile functions the model masters, mapped to native desktop handlers:
      open_wifi_settings -> ms-settings; show_map -> Maps in browser; send_email -> mailto: composer;
      create_calendar_event -> .ics (default Calendar); create_contact -> .vcf (Contacts/People).
    """
    if name in ("turn_on_flashlight", "turn_off_flashlight"):
        # no laptop flashlight — toggle the Windows night-light-ish proxy / just report
        return json.dumps({"status": "ok", "flashlight": "on" if "on" in name else "off"})
    if name == "open_wifi_settings":
        if DO_REAL:
            os.startfile("ms-settings:network-wifi")
        return json.dumps({"status": "opened", "screen": "Wi-Fi settings"})
    if name == "show_map":
        q = args.get("query", "")
        if DO_REAL and q:
            _open(f"https://www.google.com/maps/search/{quote(q)}")
        return json.dumps({"status": "opened map", "location": q})
    if name == "send_email":
        to = args.get("to", "").strip()
        subj = quote(args.get("subject", ""))
        body = quote(args.get("body", ""))
        if DO_REAL:
            _open(f"mailto:{to}?subject={subj}&body={body}")
        return json.dumps({"status": "opened email composer", "to": to, "subject": args.get("subject")})
    if name == "create_calendar_event":
        title = args.get("title", "(event)")
        dt = _ics_dt(args.get("datetime", ""))
        if DO_REAL:
            ics = ("BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//moe//agent//EN\nBEGIN:VEVENT\n"
                   f"SUMMARY:{title}\nDTSTART:{dt}\nDTEND:{dt}\nEND:VEVENT\nEND:VCALENDAR\n")
            _open(_tmp(".ics", ics))
        return json.dumps({"status": "opened calendar event", "title": title, "datetime": args.get("datetime")})
    if name == "create_contact":
        fn, ln = args.get("first_name", ""), args.get("last_name", "")
        if DO_REAL:
            vcf = ("BEGIN:VCARD\nVERSION:3.0\n"
                   f"N:{ln};{fn};;;\nFN:{fn} {ln}\n"
                   + (f"TEL;TYPE=CELL:{args['phone_number']}\n" if args.get("phone_number") else "")
                   + (f"EMAIL:{args['email']}\n" if args.get("email") else "")
                   + "END:VCARD\n")
            _open(_tmp(".vcf", vcf))
        return json.dumps({"status": "opened contact card", "name": f"{fn} {ln}".strip()})
    return json.dumps({"error": f"unknown action {name}"})


def call_api(messages):
    body = json.dumps({"model": "moe-omni-500m", "messages": messages, "tools": TOOLS,
                       "max_tokens": 120, "temperature": 0}).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    return json.load(urlopen(req, timeout=180))["choices"][0]["message"]


def agent(request):
    # SINGLE-TURN: the google/mobile-actions data is query -> all tool_calls in one message (no
    # tool-result/summary turn was ever trained). So we make ONE call, execute every call the model
    # emits (parse_calls returns them all), and stop. No agentic loop -> no drift/hallucination.
    print(f"\n=== USER: {request}")
    msg = call_api([{"role": "user", "content": request}])
    tcs = msg.get("tool_calls")
    if not tcs:
        print(f"  (no call) ASSISTANT: {(msg.get('content') or '').strip()}")
        return
    for tc in tcs:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        result = execute(name, args)
        print(f"  CALL {name}({json.dumps(args)})")
        print(f"       -> {result}")


if __name__ == "__main__":
    tasks = [a for a in sys.argv[1:] if not a.startswith("--")] or [
        "Turn on the flashlight",
        "Open my wifi settings",
        "Show me where Blue Bottle Coffee in San Francisco is on the map",
        "Add Maria Garcia to my contacts, her number is 555-0199 and email maria@acme.com",
        "Send an email to john@acme.com with the subject Lunch and tell him let's meet at noon",
        "Schedule a dentist appointment titled Dentist Checkup for 2026-06-15 at 9am",
        "Turn the flashlight off",
    ]
    print(f"mode: {'EXECUTE (real wifi/map)' if DO_REAL else 'mock'}")
    for t in tasks:
        agent(t)
