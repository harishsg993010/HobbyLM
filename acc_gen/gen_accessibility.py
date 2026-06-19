#!/usr/bin/env python3
"""Generate text-only *computer-use* tool-calling training data via Gemini.

Unlike needle (abstract function calls with no screen), every example here pairs a
serialized **accessibility tree** (a UIA-style snapshot of one window) with an
instruction, and the label is a UI action *grounded in that tree*: the model names
the target element + its control type (the `match_element` scheme we validated),
never a pixel or an index.

Output JSONL is byte-identical to needle / our `tool_data.py` single-shot format:
each line is {query, tools, answers} with `tools`/`answers` as JSON-encoded strings.
The accessibility tree lives inside `query` (prefixed `SCREEN:`), so the model is
trained as:

    TOOLS: [<12 fixed UI actions>]
    USER:  SCREEN:
           [Window] "Settings"
           [Button] "Apply"  [Edit] "Search"  ...

           turn on dark mode
    ASSISTANT: [{"name":"click","arguments":{"element":"Dark mode","control_type":"ToggleButton"}}]

The action vocabulary is FIXED (always all 12 actions are present), because at
runtime every action is always available — the model's job is selection + grounding,
not tool discovery.

Usage:
    GEMINI_API_KEY=...  python gen_accessibility.py --num-samples 5000 --workers 16
    GEMINI_API_KEY=...  python gen_accessibility.py --num-samples 20 --dry-run
"""

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from google import genai

try:
    from tqdm import tqdm
except ImportError:  # tqdm optional
    def tqdm(*a, **k):
        class _N:
            def __init__(self, *a, **k): self.n = 0
            def update(self, x=1): pass
            def close(self): pass
            def set_postfix(self, *a, **k): pass
        return _N()

MODEL = "gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# The FIXED action vocabulary (the "rich 12" the user selected). These are the
# `tools` the model picks among on every example. Schema mirrors needle's
# {name, description, parameters{param:{type,description,required}}}.
# ---------------------------------------------------------------------------
ACTIONS = [
    {"name": "click", "description": "Left-click a UI element such as a button, link, checkbox, menu item, or list item.",
     "parameters": {"element": {"type": "string", "description": "The visible name/label of the element to click, exactly as shown in the accessibility tree.", "required": True},
                    "control_type": {"type": "string", "description": "The control type of that element, e.g. 'Button', 'CheckBox', 'MenuItem', 'Hyperlink'.", "required": True}}},
    {"name": "double_click", "description": "Double-click a UI element, e.g. to open a file, folder, or list item.",
     "parameters": {"element": {"type": "string", "description": "The visible name of the element to double-click.", "required": True},
                    "control_type": {"type": "string", "description": "The control type of that element.", "required": True}}},
    {"name": "right_click", "description": "Right-click a UI element to open its context menu.",
     "parameters": {"element": {"type": "string", "description": "The visible name of the element to right-click.", "required": True},
                    "control_type": {"type": "string", "description": "The control type of that element.", "required": True}}},
    {"name": "hover", "description": "Move the pointer over a UI element to reveal a tooltip or submenu, without clicking.",
     "parameters": {"element": {"type": "string", "description": "The visible name of the element to hover over.", "required": True},
                    "control_type": {"type": "string", "description": "The control type of that element.", "required": True}}},
    {"name": "type_text", "description": "Type text into a text field, edit box, or search box.",
     "parameters": {"element": {"type": "string", "description": "The visible name/label of the field to type into.", "required": True},
                    "control_type": {"type": "string", "description": "The control type of the field, e.g. 'Edit', 'ComboBox', 'Document'.", "required": True},
                    "text": {"type": "string", "description": "The text to type, taken from the user's instruction.", "required": True}}},
    {"name": "press_key", "description": "Press a keyboard key or shortcut, e.g. 'Enter', 'Escape', 'Tab', 'Ctrl+S', 'Ctrl+C'.",
     "parameters": {"key": {"type": "string", "description": "The key or shortcut to press.", "required": True}}},
    {"name": "scroll", "description": "Scroll the current view or a scrollable element in a direction.",
     "parameters": {"direction": {"type": "string", "description": "'up', 'down', 'left', or 'right'.", "required": True},
                    "element": {"type": "string", "description": "Optional name of the scrollable element; omit to scroll the window.", "required": False}}},
    {"name": "drag", "description": "Drag one element onto another, e.g. to move a file, reorder an item, or adjust a slider.",
     "parameters": {"element": {"type": "string", "description": "The visible name of the element to drag (the source).", "required": True},
                    "control_type": {"type": "string", "description": "The control type of the source element.", "required": True},
                    "target": {"type": "string", "description": "The visible name of the element to drop onto (the destination).", "required": True}}},
    {"name": "select", "description": "Select an option from a dropdown / combo box / list, or a tab.",
     "parameters": {"element": {"type": "string", "description": "The visible name of the dropdown, list, or tab control.", "required": True},
                    "control_type": {"type": "string", "description": "The control type, e.g. 'ComboBox', 'List', 'Tab'.", "required": True},
                    "value": {"type": "string", "description": "The option/item/tab to select.", "required": True}}},
    {"name": "set_value", "description": "Set a slider, spinner, or numeric field directly to a value.",
     "parameters": {"element": {"type": "string", "description": "The visible name of the slider/spinner/field.", "required": True},
                    "control_type": {"type": "string", "description": "The control type, e.g. 'Slider', 'Spinner', 'Edit'.", "required": True},
                    "value": {"type": "number", "description": "The numeric value to set.", "required": True}}},
    {"name": "open_app", "description": "Launch an application by name when the target is not visible on the current screen.",
     "parameters": {"app_name": {"type": "string", "description": "The name of the application to open, e.g. 'Calculator', 'Notepad', 'Chrome'.", "required": True}}},
    {"name": "wait", "description": "Wait for a number of seconds for the UI to load or settle.",
     "parameters": {"seconds": {"type": "number", "description": "How many seconds to wait.", "required": True}}},
]
ACTION_NAMES = {a["name"] for a in ACTIONS}
ACTIONS_STR = json.dumps(ACTIONS, separators=(",", ":"), ensure_ascii=False)

# Actions whose `element` (and `control_type`) MUST resolve to a tree element.
_NEEDS_ELEMENT = {"click", "double_click", "right_click", "hover", "type_text", "select", "set_value", "drag"}
# Actions whose `control_type` must match the resolved element's type.
_NEEDS_CTYPE = {"click", "double_click", "right_click", "hover", "type_text", "select", "set_value", "drag"}

# Valid UIA control types (used to nudge Gemini and sanity-check).
CONTROL_TYPES = [
    "Button", "Edit", "CheckBox", "RadioButton", "ComboBox", "List", "ListItem",
    "MenuItem", "Menu", "Tab", "TabItem", "Slider", "Spinner", "Hyperlink", "Text",
    "TreeItem", "ToggleButton", "Document", "Image", "Group", "Window", "ProgressBar",
    "ScrollBar", "SplitButton", "Pane",
]

_SCROLL_DIRS = {"up", "down", "left", "right"}

# ---------------------------------------------------------------------------
# Screen archetypes — seed Gemini into many distinct Windows UI surfaces so the
# accessibility trees cover real breadth (not just one app).
# ---------------------------------------------------------------------------
ARCHETYPES = [
    "Windows Settings page (e.g. Display, Bluetooth, Network, Accounts, Privacy)",
    "File Explorer window with a folder of files and a toolbar",
    "a Save As / Open file dialog",
    "Calculator app (standard or scientific)",
    "Notepad or a plain text editor with a menu bar",
    "a web browser window with tabs, an address bar, and page content",
    "a login / sign-in dialog with username, password, and remember-me",
    "an email client composing a new message (To, Subject, Body, Send)",
    "a media player with transport controls and a track list",
    "a software installer / setup wizard (Next, Back, accept license)",
    "a multi-field registration or checkout form",
    "a spreadsheet application with a ribbon and a grid",
    "a chat / messaging app with a conversation list and message box",
    "the Windows Control Panel or a system control applet",
    "a Print dialog with printer selection and copies",
    "a Find & Replace dialog",
    "an image viewer / photo editor with editing tools",
    "a code editor / IDE with a file tree, tabs, and an editor pane",
    "a music or podcast streaming app",
    "a calendar app showing events and a new-event form",
    "a video conferencing window with mute, camera, share, and participants",
    "a PDF reader with a toolbar and page thumbnails",
    "an antivirus / system-utility dashboard with scan and settings buttons",
    "a database / admin console with a sidebar and a results table",
    "an e-commerce product page with quantity, add-to-cart, and reviews",
    "a Wi-Fi / network connection picker",
    "a date & time / region settings panel",
    "a printer or device properties multi-tab dialog",
    "a Task Manager / process list with end-task",
    "a Bluetooth device pairing dialog",
]

# Per-batch FOCUS action so the rich-12 vocabulary stays balanced across the corpus.
# Left unsteered, Gemini collapses ~60% of instructions to `click`; here each batch is
# told to make the MAJORITY of its tasks use one focus action AND to design the screen's
# elements so that action is well-supported. `click` is down-weighted (it dominates
# naturally anyway); rare actions are up-weighted so they clear a per-action floor.
_FOCUS_WEIGHTS = {
    "click": 3, "type_text": 4, "select": 4, "press_key": 4, "scroll": 4,
    "double_click": 4, "right_click": 4, "hover": 4, "set_value": 4,
    "drag": 5, "open_app": 3, "wait": 3,
    "_mix": 6,  # a natural unsteered mix
}
_FOCUS_ACTIONS = list(_FOCUS_WEIGHTS.keys())
_FOCUS_W = list(_FOCUS_WEIGHTS.values())

# Hints so Gemini builds a screen that actually supports the focus action.
_FOCUS_HINT = {
    "type_text": "include Edit/ComboBox/Document fields to type into",
    "set_value": "include at least one Slider or Spinner with a current value",
    "select": "include a ComboBox, List, or Tab control with selectable options",
    "scroll": "include a long List, Document, or Pane that can be scrolled",
    "drag": "include items that can be dragged onto a target (files, list items, or a slider)",
    "double_click": "include files, folders, or list items that are opened by double-clicking",
    "right_click": "include items whose context menu is opened by right-clicking",
    "hover": "include controls with tooltips or submenus revealed on hover",
    "press_key": "include fields/contexts where a keyboard shortcut (Enter, Ctrl+S, Tab, Delete) is natural",
    "open_app": "make some requests target apps NOT present on this screen",
    "click": "include plenty of buttons, links, checkboxes, and menu items",
}

_PROMPT = """You are generating training data for a tiny on-device computer-use model that controls a Windows PC using the ACCESSIBILITY TREE (UI Automation), not screenshots.

Produce ONE realistic screen and {n_tasks} short user instructions for it.

SCREEN ARCHETYPE for this batch: {archetype}

Return a single JSON object with this exact shape:
{{
  "window": "the window/app title",
  "elements": [
    {{"control_type": "Button", "name": "Apply", "state": "enabled"}},
    {{"control_type": "Edit", "name": "Search", "state": "empty"}}
  ],
  "tasks": [
    {{"instruction": "natural user request", "action": {{"name": "click", "arguments": {{"element": "Apply", "control_type": "Button"}}}}}}
  ]
}}

RULES FOR `elements` (the accessibility tree):
- {elements_rule}
- `control_type` MUST be one of: {ctypes}.
- `name` is the visible label/text of the control. Make names realistic and varied; some windows legitimately have two elements with the same name.
- `state` is optional, short, and realistic: "enabled", "disabled", "checked", "unchecked", "selected", "focused", "empty", or a current value like "75%". Omit if not meaningful.
- Include the kinds of controls the tasks will act on (a task can only target an element that is in this list).
{realism}
RULES FOR `tasks` (instruction -> grounded action):
- {emphasis}
- Each `instruction` is what a real user would say/type to an assistant: terse or conversational, sometimes indirect ("it's too quiet" -> raise a volume slider). Vary phrasing widely; never repeat an intent.
- The `action.name` MUST be one of: {anames}.
- GROUNDING IS CRITICAL: for click/double_click/right_click/hover/type_text/select/set_value/drag, `arguments.element` MUST be the EXACT `name` of an element you listed above, and `arguments.control_type` MUST equal that element's `control_type`. Never reference an element that is not in the list. For `drag`, `arguments.target` must also be an exact element name.
- type_text: `arguments.text` is the literal text the user wants typed, taken from THEIR words in the instruction — do not invent specifics they didn't say.
- select: `arguments.value` is the option/tab/item to choose (it may be a value not separately listed).
- set_value: `arguments.value` is a number; only for Slider/Spinner/Edit-numeric.
- press_key: `arguments.key` like "Enter", "Tab", "Escape", "Ctrl+S", "Ctrl+F", "Delete".
- scroll: `arguments.direction` is up/down/left/right; `element` optional.
- open_app: use ONLY when the user clearly wants an app that is NOT among the elements on this screen.
- wait: `arguments.seconds` is a number.
- Numbers must be JSON numbers (75 not "75"); never emit partial actions.
- {not_present}

Return ONLY the JSON object. No markdown fences, no commentary."""

_NOT_PRESENT_INSTR = ("Make about 1 in 5 instructions ask for something that is NOT present on this screen; "
                      "for those, the correct `action` is `open_app` with the relevant app_name (or, if it is a "
                      "generic scroll-to-find, a `scroll`).")
_NORMAL_INSTR = "Every instruction must be satisfiable by acting on an element that is present on this screen."

# Clean mode: small, tidy screens (the original distribution).
_ELEMENTS_CLEAN = "6 to 18 elements that realistically co-occur in this kind of window."
_REALISM_CLEAN = ""

# Realistic mode: large, cluttered trees like a raw UI-Automation dump, so the model learns to
# ground correctly WITHOUT the client pre-filtering chrome/noise. Teaches: ignore window chrome,
# ignore decorative Text/Image/Group, and disambiguate among near-identical siblings by exact name.
_ELEMENTS_REALISTIC = ("30 to 55 elements — a realistic, cluttered raw accessibility-tree dump, NOT a tidy "
                       "hand-picked list.")
_REALISM_REALISTIC = """
REALISM (make this look like a real raw UI-Automation snapshot, not a clean list):
- Include WINDOW CHROME that every real window has: a [Button] "Minimize", [Button] "Maximize", [Button] "Close" (or "Minimize <App>", "Close <App>"), an [Image] app icon, and the app title as [Text].
- Include DECORATIVE / CONTEXT elements that are NOT actionable: [Text] static labels and headings, [Group] section containers (e.g. "Standard functions", "Display controls", "Memory"), separators, status-bar [Text], a [Text] showing a current display/value. These pad the tree like real apps do.
- Include several DISABLED elements (state "disabled") mixed in.
- Include at least one CLUSTER of NEAR-IDENTICAL sibling controls where only the exact name distinguishes them — e.g. number-pad buttons "One".."Nine", a list of similar rows ("Row 1".."Row 8"), repeated "Edit"/"Delete"/"More options" buttons per list item, day cells, or seat numbers. Real UIs are full of these.
- Every element still needs a non-empty `name`.
- CRITICAL: even though the tree is cluttered, each task's target `element` must still be the EXACT name of ONE listed element, and for the near-identical clusters the instruction must make the SPECIFIC target unambiguous (e.g. "click the seven key" -> element "Seven", not "One"). Generate several tasks that target elements inside the near-identical cluster so the model learns to pick the right sibling by exact name.
"""


# ---------------------------------------------------------------------------
def make_clients():
    raw = os.environ.get("GEMINI_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        print("Error: GEMINI_API_KEY not set (comma-separate multiple keys).", file=sys.stderr)
        sys.exit(1)
    return [genai.Client(api_key=k) for k in keys]


class ClientPool:
    def __init__(self, clients):
        self._c = clients
        self._i = 0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            c = self._c[self._i % len(self._c)]
            self._i += 1
            return c


def serialize_screen(window, elements):
    """Render the element list into the canonical SCREEN block the model sees."""
    lines = [f'[Window] "{window}"']
    for e in elements:
        ct = e.get("control_type", "Text")
        nm = e.get("name", "")
        st = e.get("state")
        suffix = f"  ({st})" if st else ""
        lines.append(f'[{ct}] "{nm}"{suffix}')
    return "SCREEN:\n" + "\n".join(lines)


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def validate_task(task, name_to_types):
    """Validate one task's action against the generated tree. Returns the cleaned
    action dict, or None if it cannot be grounded."""
    if not isinstance(task, dict):
        return None
    instr = str(task.get("instruction", "")).strip()
    action = task.get("action")
    if not instr or not isinstance(action, dict):
        return None
    name = action.get("name")
    if name not in ACTION_NAMES:
        return None
    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return None
    args = dict(args)

    # Element grounding ------------------------------------------------------
    if name in _NEEDS_ELEMENT:
        el = args.get("element")
        if not isinstance(el, str) or not el.strip():
            return None
        types = name_to_types.get(_norm(el))
        if types is None:
            return None  # element not in the tree -> hallucination, drop
        if name in _NEEDS_CTYPE:
            ct = args.get("control_type")
            if not isinstance(ct, str):
                return None
            if ct not in types:
                # repair: snap to the tree's actual control type for this name
                args["control_type"] = sorted(types)[0]
        if name == "drag":
            tgt = args.get("target")
            if not isinstance(tgt, str) or _norm(tgt) not in name_to_types:
                return None

    # Per-action argument checks --------------------------------------------
    if name == "type_text":
        if not isinstance(args.get("text"), str) or not args["text"].strip():
            return None
    elif name == "select":
        if not isinstance(args.get("value"), (str, int, float)) or str(args.get("value")).strip() == "":
            return None
        args["value"] = args["value"] if isinstance(args["value"], str) else args["value"]
    elif name == "set_value":
        v = args.get("value")
        if isinstance(v, str):
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                return None
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        args["value"] = v
    elif name == "press_key":
        if not isinstance(args.get("key"), str) or not args["key"].strip():
            return None
    elif name == "scroll":
        d = str(args.get("direction", "")).lower()
        if d not in _SCROLL_DIRS:
            return None
        args["direction"] = d
        if "element" in args and (not isinstance(args["element"], str) or _norm(args["element"]) not in name_to_types):
            args.pop("element", None)  # ungrounded optional scroll element -> drop it
    elif name == "open_app":
        if not isinstance(args.get("app_name"), str) or not args["app_name"].strip():
            return None
    elif name == "wait":
        s = args.get("seconds")
        if isinstance(s, str):
            try:
                s = float(s) if "." in s else int(s)
            except ValueError:
                return None
        if not isinstance(s, (int, float)) or isinstance(s, bool):
            return None
        args["seconds"] = s

    return {"name": name, "arguments": args}


def generate_batch(pool, rng, model, n_tasks=12, realistic_frac=0.0):
    archetype = rng.choice(ARCHETYPES)
    realistic = rng.random() < realistic_frac
    focus = rng.choices(_FOCUS_ACTIONS, weights=_FOCUS_W, k=1)[0]
    if focus == "_mix":
        emphasis = "Use a natural mix across all action types; vary the action from task to task."
    else:
        n_focus = max(2, (n_tasks * 2) // 3)
        hint = _FOCUS_HINT.get(focus)
        hint_str = f" Design the screen so this is natural: {hint}." if hint else ""
        emphasis = (f"At least {n_focus} of the {n_tasks} instructions MUST use the `{focus}` action.{hint_str} "
                    f"Let the remaining instructions use other actions naturally.")
    not_present = _NOT_PRESENT_INSTR if (focus == "open_app" or rng.random() < 0.30) else _NORMAL_INSTR
    prompt = _PROMPT.format(
        n_tasks=n_tasks, archetype=archetype, emphasis=emphasis,
        ctypes=", ".join(CONTROL_TYPES), anames=", ".join(sorted(ACTION_NAMES)),
        not_present=not_present,
        elements_rule=(_ELEMENTS_REALISTIC if realistic else _ELEMENTS_CLEAN),
        realism=(_REALISM_REALISTIC if realistic else _REALISM_CLEAN),
    )
    temperature = rng.choice([0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2])
    client = pool.get()
    try:
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config={"temperature": temperature,
                    "max_output_tokens": 16384 if realistic else 8192,  # bigger trees need more room
                    "response_mime_type": "application/json"},
        )
    except Exception as e:
        print(f"  gemini error: {str(e)[:120]}", file=sys.stderr)
        return []

    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []

    window = str(obj.get("window", "")).strip() or "Window"
    elements = obj.get("elements")
    tasks = obj.get("tasks")
    if not isinstance(elements, list) or not isinstance(tasks, list) or not elements:
        return []

    # Build name -> set(control_types) map from the tree.
    name_to_types = {}
    clean_elements = []
    for e in elements:
        if not isinstance(e, dict):
            continue
        nm = e.get("name")
        ct = e.get("control_type")
        if not isinstance(nm, str) or not nm.strip() or ct not in CONTROL_TYPES:
            continue
        clean_elements.append(e)
        name_to_types.setdefault(_norm(nm), set()).add(ct)
    if not clean_elements:
        return []

    screen_str = serialize_screen(window, clean_elements)

    out = []
    for t in tasks:
        cleaned = validate_task(t, name_to_types)
        if cleaned is None:
            continue
        instr = str(t["instruction"]).strip()
        query = f"{screen_str}\n\n{instr}"
        out.append({
            "query": query,
            "tools": ACTIONS_STR,
            "answers": json.dumps([cleaned], separators=(",", ":"), ensure_ascii=False),
            "source": "synth-gemini-computeruse",
            "model": model,
            "archetype": archetype,
            "action": cleaned["name"],
            "window": window,
            "realistic": realistic,
            "n_elements": len(clean_elements),
        })
    return out


def generate_all(num_samples, workers, model, pool, n_tasks=12, seed=42, realistic_frac=0.0):
    """Submit batches in waves so we stop calling Gemini as soon as we hit the
    target (rather than pre-submitting every batch and burning quota on dedup loss)."""
    rng = random.Random(seed)
    results = []
    seen = set()
    pbar = tqdm(total=num_samples, desc="Generating", unit="ex")
    submitted = 0
    max_batches = max(8, int(num_samples / max(1, n_tasks) * 3.0) + 8)  # safety cap

    def _one(_i):
        brng = random.Random(rng.randint(0, 2**32 - 1))
        return generate_batch(pool, brng, model, n_tasks=n_tasks, realistic_frac=realistic_frac)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        while len(results) < num_samples and submitted < max_batches:
            # Estimate how many more batches we need, then submit a wave of that
            # many (bounded by the worker count) so in-flight calls overlap.
            remaining = num_samples - len(results)
            need = max(1, -(-remaining // max(1, n_tasks // 2)))  # assume ~half survive dedup/validation
            wave = min(workers, need, max_batches - submitted)
            futures = [ex.submit(_one, submitted + i) for i in range(wave)]
            submitted += wave
            for fut in futures:
                for r in fut.result():
                    key = _norm(r["window"]) + "||" + _norm(r["query"].split("\n\n")[-1])
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(results) >= num_samples:
                        break
                    results.append(r)
                    pbar.update(1)
    pbar.close()
    if len(results) < num_samples:
        print(f"  note: hit safety cap of {max_batches} batches; got {len(results)}/{num_samples}.", file=sys.stderr)
    return results[:num_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks-per-screen", type=int, default=12)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="acc_computeruse.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="90/5/5",
                    help="train/val/test percentage split, e.g. 90/5/5. Set to '' for a single file.")
    ap.add_argument("--dry-run", action="store_true", help="print examples, do not write file")
    args = ap.parse_args()

    pool = ClientPool(make_clients())
    rows = generate_all(args.num_samples, args.workers, args.model, pool,
                        n_tasks=args.tasks_per_screen, seed=args.seed)

    # Report action distribution.
    from collections import Counter
    dist = Counter(r["action"] for r in rows)
    print(f"\nGenerated {len(rows)} examples. Action distribution:", file=sys.stderr)
    for a, c in dist.most_common():
        print(f"  {a:14s} {c}", file=sys.stderr)

    if args.dry_run:
        for r in rows[:8]:
            print("\n" + "=" * 70)
            print(r["query"])
            print("ANSWER:", r["answers"])
        return

    def _write(path, recs):
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not args.split.strip():
        _write(args.out, rows)
        print(f"Wrote {len(rows)} -> {args.out}", file=sys.stderr)
        return

    # Stratify by action so every split keeps the same action distribution, then
    # shuffle deterministically and slice 90/5/5 (or whatever was requested).
    pcts = [float(x) for x in args.split.split("/")]
    if len(pcts) != 3 or abs(sum(pcts) - 100) > 1e-6:
        print(f"Error: --split must be three numbers summing to 100 (got {args.split}).", file=sys.stderr)
        sys.exit(1)
    srng = random.Random(args.seed)
    by_action = {}
    for r in rows:
        by_action.setdefault(r["action"], []).append(r)
    train, val, test = [], [], []
    for action, recs in by_action.items():
        srng.shuffle(recs)
        n = len(recs)
        n_tr = round(n * pcts[0] / 100)
        n_va = round(n * pcts[1] / 100)
        train += recs[:n_tr]
        val += recs[n_tr:n_tr + n_va]
        test += recs[n_tr + n_va:]
    for s in (train, val, test):
        srng.shuffle(s)

    base = args.out[:-6] if args.out.endswith(".jsonl") else args.out
    paths = {f"{base}.train.jsonl": train, f"{base}.val.jsonl": val, f"{base}.test.jsonl": test}
    for p, recs in paths.items():
        _write(p, recs)
    print(f"Split {len(rows)} -> train {len(train)} / val {len(val)} / test {len(test)} "
          f"(stratified by action)", file=sys.stderr)
    for p in paths:
        print(f"  {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
