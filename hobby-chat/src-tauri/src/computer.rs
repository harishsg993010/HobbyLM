//! Native computer-use. Snapshots the live Windows UI-Automation (accessibility) tree of a target
//! window, serializes it exactly like the model's training data (`SCREEN:\n[Ctype] "Name" (state)`),
//! asks the in-process 500M model for ONE grounded action (the 12 fixed UI actions), grounds the
//! model's {element,control_type} back to a live element by fuzzy match, and executes it via UIA
//! control patterns. The model picks the verb + names the element; nothing is hardcoded.

use hobby_rs::{Engine, GenOpts};
use serde_json::{json, Value};
use std::sync::Arc;
use uiautomation::patterns::{UIInvokePattern, UIRangeValuePattern, UISelectionItemPattern, UIValuePattern};
use uiautomation::types::ControlType;
use uiautomation::{UIAutomation, UIElement, UITreeWalker};

/// The planning vocabulary = the 12 grounding actions + `finish` (byte-identical to v4 training's
/// ACTIONS_PLAN). Built by appending the finish tool to the 12-action schema.
fn actions_plan() -> String {
    let finish = r#"{"name":"finish","description":"Signal that the goal has been fully achieved; no more actions are needed.","parameters":{}}"#;
    format!("{},{}]", &ACTIONS[..ACTIONS.len() - 1], finish)
}

/// Reconstruct the trained-style atomic instruction for the DONE-history (v4 was trained on natural
/// phrasings like "click the Eight button", not terse "click Eight").
fn instr_for(name: &str, args: &Value) -> String {
    let el = arg_str(args, "element");
    match name {
        "click" => format!("click the {el} button"),
        "double_click" => format!("double-click the {el} button"),
        "right_click" => format!("right-click the {el} button"),
        "hover" => format!("hover over the {el} button"),
        "type_text" => format!("type {} into the {el} field", arg_str(args, "text")),
        "press_key" => format!("press {}", arg_str(args, "key")),
        "scroll" => format!("scroll {}", arg_str(args, "direction")),
        "select" => format!("select {} in {el}", arg_str(args, "value")),
        "set_value" => format!("set {el} to {}", args.get("value").map(|v| v.to_string()).unwrap_or_default()),
        _ => format!("{name} {el}"),
    }
}

/// The fixed 12-action vocabulary, byte-identical to the training `TOOLS:` schema (flat params).
const ACTIONS: &str = r#"[{"name":"click","description":"Left-click a UI element such as a button, link, checkbox, menu item, or list item.","parameters":{"element":{"type":"string","description":"The visible name/label of the element to click, exactly as shown in the accessibility tree.","required":true},"control_type":{"type":"string","description":"The control type of that element, e.g. 'Button', 'CheckBox', 'MenuItem', 'Hyperlink'.","required":true}}},{"name":"double_click","description":"Double-click a UI element, e.g. to open a file, folder, or list item.","parameters":{"element":{"type":"string","description":"The visible name of the element to double-click.","required":true},"control_type":{"type":"string","description":"The control type of that element.","required":true}}},{"name":"right_click","description":"Right-click a UI element to open its context menu.","parameters":{"element":{"type":"string","description":"The visible name of the element to right-click.","required":true},"control_type":{"type":"string","description":"The control type of that element.","required":true}}},{"name":"hover","description":"Move the pointer over a UI element to reveal a tooltip or submenu, without clicking.","parameters":{"element":{"type":"string","description":"The visible name of the element to hover over.","required":true},"control_type":{"type":"string","description":"The control type of that element.","required":true}}},{"name":"type_text","description":"Type text into a text field, edit box, or search box.","parameters":{"element":{"type":"string","description":"The visible name/label of the field to type into.","required":true},"control_type":{"type":"string","description":"The control type of the field, e.g. 'Edit', 'ComboBox', 'Document'.","required":true},"text":{"type":"string","description":"The text to type, taken from the user's instruction.","required":true}}},{"name":"press_key","description":"Press a keyboard key or shortcut, e.g. 'Enter', 'Escape', 'Tab', 'Ctrl+S', 'Ctrl+C'.","parameters":{"key":{"type":"string","description":"The key or shortcut to press.","required":true}}},{"name":"scroll","description":"Scroll the current view or a scrollable element in a direction.","parameters":{"direction":{"type":"string","description":"'up', 'down', 'left', or 'right'.","required":true},"element":{"type":"string","description":"Optional name of the scrollable element; omit to scroll the window.","required":false}}},{"name":"drag","description":"Drag one element onto another, e.g. to move a file, reorder an item, or adjust a slider.","parameters":{"element":{"type":"string","description":"The visible name of the element to drag (the source).","required":true},"control_type":{"type":"string","description":"The control type of the source element.","required":true},"target":{"type":"string","description":"The visible name of the element to drop onto (the destination).","required":true}}},{"name":"select","description":"Select an option from a dropdown / combo box / list, or a tab.","parameters":{"element":{"type":"string","description":"The visible name of the dropdown, list, or tab control.","required":true},"control_type":{"type":"string","description":"The control type, e.g. 'ComboBox', 'List', 'Tab'.","required":true},"value":{"type":"string","description":"The option/item/tab to select.","required":true}}},{"name":"set_value","description":"Set a slider, spinner, or numeric field directly to a value.","parameters":{"element":{"type":"string","description":"The visible name of the slider/spinner/field.","required":true},"control_type":{"type":"string","description":"The control type, e.g. 'Slider', 'Spinner', 'Edit'.","required":true},"value":{"type":"number","description":"The numeric value to set.","required":true}}},{"name":"open_app","description":"Launch an application by name when the target is not visible on the current screen.","parameters":{"app_name":{"type":"string","description":"The name of the application to open, e.g. 'Calculator', 'Notepad', 'Chrome'.","required":true}}},{"name":"wait","description":"Wait for a number of seconds for the UI to load or settle.","parameters":{"seconds":{"type":"number","description":"How many seconds to wait.","required":true}}}]"#;

const MAX_ELEMENTS: usize = 45;
const CAP_WALK: usize = 200;

/// UIA control types kept in the snapshot (interactive + a little context), matching harvest_real.py.
fn ctype_name(ct: ControlType) -> &'static str {
    match ct {
        ControlType::Button => "Button",
        ControlType::Edit => "Edit",
        ControlType::CheckBox => "CheckBox",
        ControlType::RadioButton => "RadioButton",
        ControlType::ComboBox => "ComboBox",
        ControlType::List => "List",
        ControlType::ListItem => "ListItem",
        ControlType::MenuItem => "MenuItem",
        ControlType::Tab => "Tab",
        ControlType::TabItem => "TabItem",
        ControlType::Slider => "Slider",
        ControlType::Spinner => "Spinner",
        ControlType::Hyperlink => "Hyperlink",
        ControlType::TreeItem => "TreeItem",
        ControlType::SplitButton => "SplitButton",
        ControlType::Text => "Text",
        ControlType::Image => "Image",
        ControlType::Group => "Group",
        ControlType::Document => "Document",
        _ => "Other",
    }
}

fn is_interactive(ct: &str) -> bool {
    matches!(ct, "Button" | "Edit" | "CheckBox" | "RadioButton" | "ComboBox" | "List" | "ListItem"
        | "MenuItem" | "Tab" | "TabItem" | "Slider" | "Spinner" | "Hyperlink" | "TreeItem" | "SplitButton")
}

struct El {
    ctype: &'static str,
    name: String,
    state: String,
    el: UIElement,
}

fn el_state(el: &UIElement, ctype: &str) -> String {
    if !el.is_enabled().unwrap_or(true) {
        return "disabled".into();
    }
    if let Ok(rv) = el.get_pattern::<UIRangeValuePattern>() {
        if let Ok(v) = rv.get_value() {
            return format!("{}", v as i64);
        }
    }
    let _ = ctype;
    "enabled".into()
}

fn collect(el: &UIElement, out: &mut Vec<El>, seen: &mut Vec<(String, String)>) {
    let ct = match el.get_control_type() {
        Ok(c) => ctype_name(c),
        Err(_) => return,
    };
    if ct == "Other" {
        return;
    }
    let name = el.get_name().unwrap_or_default().trim().to_string();
    if name.is_empty() || el.is_offscreen().unwrap_or(true) {
        return;
    }
    if let Ok(r) = el.get_bounding_rectangle() {
        if r.get_right() - r.get_left() <= 2 || r.get_bottom() - r.get_top() <= 2 {
            return;
        }
    }
    let key = (ct.to_string(), name.to_lowercase());
    if seen.contains(&key) {
        return;
    }
    seen.push(key);
    let state = el_state(el, ct);
    out.push(El { ctype: ct, name, state, el: el.clone() });
}

fn walk(walker: &UITreeWalker, el: &UIElement, out: &mut Vec<El>, seen: &mut Vec<(String, String)>, depth: u32) {
    if depth > 32 || out.len() >= CAP_WALK {
        return;
    }
    if let Ok(child) = walker.get_first_child(el) {
        let mut cur = child;
        loop {
            collect(&cur, out, seen);
            walk(walker, &cur, out, seen, depth + 1);
            match walker.get_next_sibling(&cur) {
                Ok(n) => cur = n,
                Err(_) => break,
            }
            if out.len() >= CAP_WALK {
                break;
            }
        }
    }
}

/// Cap to the model's context budget, interactive controls first, preserving original order.
fn cap(els: Vec<El>) -> Vec<El> {
    if els.len() <= MAX_ELEMENTS {
        return els;
    }
    let mut keep_idx: Vec<usize> = (0..els.len()).filter(|&i| is_interactive(els[i].ctype)).collect();
    if keep_idx.len() > MAX_ELEMENTS {
        keep_idx.truncate(MAX_ELEMENTS);
    } else {
        for i in 0..els.len() {
            if keep_idx.len() >= MAX_ELEMENTS {
                break;
            }
            if !is_interactive(els[i].ctype) {
                keep_idx.push(i);
            }
        }
    }
    keep_idx.sort_unstable();
    let keep: std::collections::HashSet<usize> = keep_idx.into_iter().collect();
    els.into_iter().enumerate().filter(|(i, _)| keep.contains(i)).map(|(_, e)| e).collect()
}

fn serialize(title: &str, els: &[El]) -> String {
    let mut s = format!("SCREEN:\n[Window] \"{}\"", title);
    for e in els {
        let st = if e.state.is_empty() { String::new() } else { format!("  ({})", e.state) };
        s.push_str(&format!("\n[{}] \"{}\"{}", e.ctype, e.name, st));
    }
    s
}

fn keys(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !matches!(*w, "the" | "a" | "an" | "of" | "to" | "button" | "field" | "box"))
        .map(|w| w.to_string())
        .collect()
}

/// Best element whose name tokens overlap the model's name; control_type match is a bonus.
fn match_element(name: &str, ctype: &str, els: &[El]) -> Option<usize> {
    let want = keys(name);
    let (mut best, mut score) = (None, 0i32);
    for (i, e) in els.iter().enumerate() {
        let ek = keys(&e.name);
        let mut s = 0i32;
        for k in &ek {
            if want.contains(k) {
                s += k.len() as i32 + 2;
            }
        }
        if e.name.trim().eq_ignore_ascii_case(name.trim()) {
            s += 20;
        }
        if !ctype.is_empty() && e.ctype.eq_ignore_ascii_case(ctype) {
            s += 3;
        }
        if s > score {
            best = Some(i);
            score = s;
        }
    }
    if score >= 3 {
        best
    } else {
        None
    }
}

fn find_window(automation: &UIAutomation, name: &str) -> Result<UIElement, String> {
    automation
        .create_matcher()
        .contains_name(name)
        .depth(3)
        .find_first()
        .map_err(|e| format!("window '{name}' not found: {e}"))
}

fn key_to_sendkeys(k: &str) -> String {
    let (mut prefix, mut main) = (String::new(), String::new());
    for p in k.split(['+', '-']).map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match p.to_lowercase().as_str() {
            "ctrl" | "control" => prefix.push('^'),
            "alt" => prefix.push('%'),
            "shift" => prefix.push('+'),
            "enter" | "return" => main = "{Enter}".into(),
            "tab" => main = "{Tab}".into(),
            "esc" | "escape" => main = "{Esc}".into(),
            "delete" | "del" => main = "{Delete}".into(),
            "backspace" => main = "{Back}".into(),
            "space" => main = "{Space}".into(),
            "up" => main = "{Up}".into(),
            "down" => main = "{Down}".into(),
            "left" => main = "{Left}".into(),
            "right" => main = "{Right}".into(),
            "home" => main = "{Home}".into(),
            "end" => main = "{End}".into(),
            other => main = if other.len() == 1 { other.to_string() } else { format!("{{{other}}}") },
        }
    }
    format!("{prefix}{main}")
}

fn arg_str<'a>(args: &'a Value, k: &str) -> &'a str {
    args.get(k).and_then(|v| v.as_str()).unwrap_or("")
}

/// Resolve + (optionally) perform the action. Returns a human-readable log line.
fn act(automation: &UIAutomation, win: &UIElement, name: &str, args: &Value, els: &[El], execute: bool) -> String {
    let resolve = |nm: &str, ct: &str| match_element(nm, ct, els).map(|i| &els[i]);
    let el_arg = arg_str(args, "element");
    let ct_arg = arg_str(args, "control_type");

    match name {
        "click" | "double_click" | "right_click" | "hover" => {
            let Some(e) = resolve(el_arg, ct_arg) else { return format!("UNRESOLVED element {el_arg:?}") };
            let plan = format!("{name} -> [{}] \"{}\"", e.ctype, e.name);
            if !execute {
                return plan;
            }
            let r = match name {
                // Prefer programmatic Invoke/Select — works regardless of window z-order/focus, so the
                // click lands on the target even when hobby-chat is the foreground window. Fall back to a
                // real mouse click for controls that expose neither pattern.
                "click" => e.el.get_pattern::<UIInvokePattern>().and_then(|p| p.invoke())
                    .or_else(|_| e.el.get_pattern::<UISelectionItemPattern>().and_then(|p| p.select()))
                    .or_else(|_| e.el.click()),
                "double_click" => e.el.double_click(),
                "right_click" => e.el.right_click(),
                _ => e.el.set_focus(), // hover ≈ focus
            };
            match r {
                Ok(_) => format!("OK {plan}"),
                Err(err) => format!("FAILED {plan}: {err}"),
            }
        }
        "type_text" => {
            let Some(e) = resolve(el_arg, ct_arg) else { return format!("UNRESOLVED field {el_arg:?}") };
            let text = arg_str(args, "text");
            let plan = format!("type {text:?} into [{}] \"{}\"", e.ctype, e.name);
            if !execute {
                return plan;
            }
            if let Ok(vp) = e.el.get_pattern::<UIValuePattern>() {
                return match vp.set_value(text) {
                    Ok(_) => format!("OK {plan}"),
                    Err(err) => format!("FAILED {plan}: {err}"),
                };
            }
            let _ = e.el.try_focus();
            match e.el.send_keys(text, 5) {
                Ok(_) => format!("OK {plan} (keys)"),
                Err(err) => format!("FAILED {plan}: {err}"),
            }
        }
        "set_value" => {
            let Some(e) = resolve(el_arg, ct_arg) else { return format!("UNRESOLVED {el_arg:?}") };
            let v = args.get("value").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let plan = format!("set [{}] \"{}\" = {v}", e.ctype, e.name);
            if !execute {
                return plan;
            }
            if let Ok(rv) = e.el.get_pattern::<UIRangeValuePattern>() {
                return match rv.set_value(v) {
                    Ok(_) => format!("OK {plan}"),
                    Err(err) => format!("FAILED {plan}: {err}"),
                };
            }
            if let Ok(vp) = e.el.get_pattern::<UIValuePattern>() {
                return match vp.set_value(&format!("{v}")) {
                    Ok(_) => format!("OK {plan} (value)"),
                    Err(err) => format!("FAILED {plan}: {err}"),
                };
            }
            format!("UNSUPPORTED {plan}")
        }
        "press_key" => {
            let key = arg_str(args, "key");
            let sk = key_to_sendkeys(key);
            let plan = format!("press {key:?}");
            if !execute {
                return plan;
            }
            match win.send_keys(&sk, 5) {
                Ok(_) => format!("OK {plan}"),
                Err(err) => format!("FAILED {plan}: {err}"),
            }
        }
        "open_app" => {
            let app = arg_str(args, "app_name");
            let plan = format!("open_app {app:?}");
            if !execute {
                return plan;
            }
            match std::process::Command::new("cmd").args(["/c", "start", "", app]).spawn() {
                Ok(_) => format!("OK {plan}"),
                Err(err) => format!("FAILED {plan}: {err}"),
            }
        }
        "wait" => {
            let s = args.get("seconds").and_then(|x| x.as_f64()).unwrap_or(1.0).min(5.0);
            if execute {
                std::thread::sleep(std::time::Duration::from_secs_f64(s));
            }
            format!("wait {s}s")
        }
        "scroll" | "drag" | "select" => {
            // Resolve for display; these are preview-only in v1 (executed via the Python harness).
            let _ = automation;
            let tgt = resolve(el_arg, ct_arg).map(|e| e.name.clone()).unwrap_or_else(|| el_arg.to_string());
            format!("{name} {tgt:?} (preview only — not auto-executed in-app)")
        }
        other => format!("(unhandled action {other})"),
    }
}

/// UI Automation must run on a thread whose COM apartment we control — the Tauri command thread
/// already has COM initialized (by the WebView2 runtime) in a mode that conflicts with what the
/// uiautomation crate wants ("Cannot change thread mode after it is set"). Running on a fresh thread
/// lets `UIAutomation::new()` CoInitialize cleanly. Everything (snapshot + model + execute) happens
/// on that thread; only the Send result crosses back.
pub fn list_windows() -> Result<Vec<String>, String> {
    std::thread::spawn(list_windows_inner)
        .join()
        .unwrap_or_else(|_| Err("UIA thread panicked".into()))
}

fn list_windows_inner() -> Result<Vec<String>, String> {
    let automation = UIAutomation::new().map_err(|e| e.to_string())?;
    let root = automation.get_root_element().map_err(|e| e.to_string())?;
    let walker = automation.get_control_view_walker().map_err(|e| e.to_string())?;
    let mut out: Vec<String> = Vec::new();
    if let Ok(child) = walker.get_first_child(&root) {
        let mut cur = child;
        loop {
            if let Ok(n) = cur.get_name() {
                let n = n.trim().to_string();
                if !n.is_empty() && !out.contains(&n) && !n.eq_ignore_ascii_case("hobby-chat") {
                    out.push(n);
                }
            }
            match walker.get_next_sibling(&cur) {
                Ok(s) => cur = s,
                Err(_) => break,
            }
        }
    }
    Ok(out)
}

/// Snapshot `window`, ask the model for one grounded action, and run (or preview) it.
pub fn run(engine: Arc<Engine>, instruction: &str, window: &str, execute: bool) -> Result<Value, String> {
    let (instruction, window) = (instruction.to_string(), window.to_string());
    std::thread::spawn(move || run_inner(engine, &instruction, &window, execute))
        .join()
        .unwrap_or_else(|_| Err("UIA thread panicked".into()))
}

fn run_inner(engine: Arc<Engine>, instruction: &str, window: &str, execute: bool) -> Result<Value, String> {
    let automation = UIAutomation::new().map_err(|e| e.to_string())?;
    let win = find_window(&automation, window)?;
    let _ = win.try_focus();
    let walker = automation.get_control_view_walker().map_err(|e| e.to_string())?;

    let mut els = Vec::new();
    let mut seen = Vec::new();
    walk(&walker, &win, &mut els, &mut seen, 0);
    let els = cap(els);
    if els.is_empty() {
        return Err("no actionable elements found in that window".into());
    }
    let title = win.get_name().unwrap_or_else(|_| window.to_string());
    let screen = serialize(&title, &els);

    let prompt = format!("TOOLS: {ACTIONS}\nUSER: {screen}\n\n{instruction}\nASSISTANT: [");
    let opts = GenOpts { max_new: 96, temp: 0.0, top_p: 1.0, seed: 1234, rep_penalty: 1.0 };
    let mut out = String::from("[");
    engine.generate(&prompt, &[], &opts, |p| {
        out.push_str(p);
        true
    });

    let calls = crate::tools::parse_calls(&out);
    let (name, args) = calls
        .into_iter()
        .next()
        .ok_or_else(|| "model produced no parseable action".to_string())?;
    let result = act(&automation, &win, &name, &args, &els, execute);

    Ok(json!({
        "window": title,
        "n_elements": els.len(),
        "screen": screen,
        "action": { "name": name, "arguments": args },
        "executed": execute,
        "result": result,
    }))
}

/// PLANNING mode (v4): given a high-level GOAL, repeatedly snapshot -> ask the model for the next
/// grounded action (planning format, 13-action vocab incl `finish`) -> execute -> re-snapshot, until
/// the model emits `finish` or max_steps. Runs on a dedicated thread for clean COM (see run()).
pub fn run_goal(engine: Arc<Engine>, goal: &str, window: &str, max_steps: usize, execute: bool) -> Result<Value, String> {
    let (goal, window) = (goal.to_string(), window.to_string());
    std::thread::spawn(move || run_goal_inner(engine, &goal, &window, max_steps, execute))
        .join()
        .unwrap_or_else(|_| Err("UIA thread panicked".into()))
}

fn run_goal_inner(engine: Arc<Engine>, goal: &str, window: &str, max_steps: usize, execute: bool) -> Result<Value, String> {
    let automation = UIAutomation::new().map_err(|e| e.to_string())?;
    let walker = automation.get_control_view_walker().map_err(|e| e.to_string())?;
    let tools = actions_plan();
    let mut history: Vec<String> = Vec::new();
    let mut steps_log: Vec<Value> = Vec::new();

    for step in 1..=max_steps {
        let win = find_window(&automation, window)?;
        let _ = win.try_focus();
        let mut els = Vec::new();
        let mut seen = Vec::new();
        walk(&walker, &win, &mut els, &mut seen, 0);
        let els = cap(els);
        if els.is_empty() {
            return Err("no actionable elements found in that window".into());
        }
        let title = win.get_name().unwrap_or_else(|_| window.to_string());
        let screen = serialize(&title, &els);
        let done = if history.is_empty() { "nothing yet".to_string() } else { history.join("; ") };

        let prompt = format!("TOOLS: {tools}\nUSER: {screen}\n\nGOAL: {goal}\nDONE: {done}\nNEXT:\nASSISTANT: [");
        let opts = GenOpts { max_new: 64, temp: 0.0, top_p: 1.0, seed: 1234, rep_penalty: 1.0 };
        let mut out = String::from("[");
        engine.generate(&prompt, &[], &opts, |p| {
            out.push_str(p);
            true
        });

        let calls = crate::tools::parse_calls(&out);
        let Some((name, args)) = calls.into_iter().next() else {
            steps_log.push(json!({"step": step, "action": Value::Null, "result": "model produced no action — stopping"}));
            break;
        };
        if name == "finish" {
            steps_log.push(json!({"step": step, "action": {"name": "finish"}, "result": "goal complete"}));
            break;
        }
        let _ = win.set_focus(); // raise the target so any fallback mouse click lands on it, not hobby-chat
        let result = act(&automation, &win, &name, &args, &els, execute);
        let failed = result.starts_with("UNRESOLVED") || result.starts_with("FAILED");
        steps_log.push(json!({"step": step, "action": {"name": name, "arguments": args}, "result": result}));
        if failed {
            break;
        }
        history.push(instr_for(&name, &args));
        if execute {
            std::thread::sleep(std::time::Duration::from_millis(450));
        }
    }

    Ok(json!({ "goal": goal, "window": window, "executed": execute, "steps": steps_log }))
}
