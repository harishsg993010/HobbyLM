//! Function-calling tools: the JSON schema given to the model, a lenient parser for the
//! model's `{"name":..,"arguments":{..}}` output, and the executors.

use serde_json::{json, Value};

/// The tool schema list injected into the prompt as `TOOLS: [...]` (compact, no example values
/// in descriptions — this 500M model copies them into its arguments otherwise). The two built-ins
/// are followed by any tools exposed by connected MCP servers.
pub fn schema_json(mcp_tools: &[crate::mcp::McpTool]) -> String {
    let mut arr = vec![
        json!({
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }
        }),
        json!({
            "name": "calculate",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": { "expression": { "type": "string" } },
                "required": ["expression"]
            }
        }),
    ];
    for t in mcp_tools {
        // Compact the MCP schema: long descriptions + per-property docs overwhelm the 500M model.
        // Keep the tool name, a short description, and just property names + types.
        let slim_props: serde_json::Map<String, Value> = t
            .schema
            .get("properties")
            .and_then(|p| p.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| {
                        let ty = v.get("type").cloned().unwrap_or_else(|| json!("string"));
                        (k.clone(), json!({ "type": ty }))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let required = t.schema.get("required").cloned().unwrap_or_else(|| json!([]));
        let desc: String = t.description.chars().take(80).collect();
        arr.push(json!({
            "name": t.name,
            "description": desc,
            "parameters": { "type": "object", "properties": slim_props, "required": required },
        }));
    }
    Value::Array(arr).to_string()
}

/// Normalize a model-emitted tool name (it adds stray spaces / casing): trim + lowercase.
pub fn norm_name(name: &str) -> String {
    name.trim().to_lowercase()
}

/// True if `name` is one of the built-in tools (tolerant of the model's casing/whitespace).
pub fn is_builtin(name: &str) -> bool {
    matches!(norm_name(name).as_str(), "get_weather" | "calculate")
}

/// Recursively find the first string value for `key` (the model sometimes nests arguments).
fn find_str(v: &Value, key: &str) -> Option<String> {
    match v {
        Value::Object(m) => {
            if let Some(Value::String(s)) = m.get(key) {
                return Some(s.clone());
            }
            for (_, val) in m {
                if let Some(s) = find_str(val, key) {
                    return Some(s);
                }
            }
            None
        }
        Value::Array(a) => a.iter().find_map(|x| find_str(x, key)),
        _ => None,
    }
}

/// Lenient extraction of the first `{"name":..,"arguments":{..}}` object from model output.
pub fn parse_call(text: &str) -> Option<(String, Value)> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0i32;
    let mut end = None;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes[start..].iter().enumerate() {
        let c = b as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    let obj = &text[start..end?];
    let v: Value = serde_json::from_str(obj).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    // the model often copies the schema key — accept arguments / parameters / args
    let args = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("args"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    Some((name, args))
}

/// Extract ALL `{"name":..,"arguments":{..}}` calls from the model output (the mobile-actions data is
/// parallel — the model emits `[{call1},{call2},..]`). Brace-balanced, string-aware; accepts
/// arguments / parameters / args as the args key.
pub fn parse_calls(text: &str) -> Vec<(String, Value)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let (mut depth, mut start, mut in_str, mut esc) = (0i32, None, false, false);
    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start.take() {
                        if let Ok(v) = serde_json::from_str::<Value>(&text[s..=i]) {
                            if let Some(name) = v.get("name").and_then(|n| n.as_str()) {
                                let args = v
                                    .get("arguments")
                                    .or_else(|| v.get("parameters"))
                                    .or_else(|| v.get("args"))
                                    .cloned()
                                    .unwrap_or_else(|| json!({}));
                                out.push((name.to_string(), args));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Execute a tool and return a JSON string result (fed back to the model as `TOOL: ...`).
pub fn execute(name: &str, args: &Value) -> String {
    match norm_name(name).as_str() {
        "get_weather" => {
            let loc = find_str(args, "location").unwrap_or_else(|| "unknown".to_string());
            // deterministic mock weather (no network)
            let temp = 12 + (loc.bytes().map(|b| b as u32).sum::<u32>() % 18); // 12..30 C
            let conds = ["sunny", "partly cloudy", "overcast", "light rain"];
            let cond = conds[loc.len() % conds.len()];
            json!({ "location": loc, "temperature_c": temp, "condition": cond }).to_string()
        }
        "calculate" => {
            let expr = find_str(args, "expression").unwrap_or_default();
            match evalexpr::eval(&expr) {
                Ok(v) => json!({ "result": v.to_string() }).to_string(),
                Err(e) => json!({ "error": e.to_string() }).to_string(),
            }
        }
        _ => json!({ "error": format!("unknown tool {name}") }).to_string(),
    }
}
