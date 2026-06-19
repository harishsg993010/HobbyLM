//! Minimal OpenAI-compatible HTTP server on 127.0.0.1:11250, backed by the app's loaded Engine.
//! Lets any OpenAI client (python `openai`, curl, LangChain, …) talk to the local MoE. Hand-rolled
//! HTTP/1.1 (dependency-light) with SSE streaming. Endpoints: GET /v1/models, POST /v1/chat/completions.

use crate::encoders::Encoders;
use base64::Engine as _;
use hobby_rs::{Engine, GenOpts};
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

type EngineSlot = Arc<Mutex<Option<Arc<Engine>>>>;
type EncodersSlot = Arc<Mutex<Encoders>>;
const MODEL_ID: &str = "moe-omni-500m";
static SEQ: AtomicU64 = AtomicU64::new(1);

/// Start the server on a background thread. Non-fatal if the port is taken.
pub fn spawn(engine: EngineSlot, encoders: EncodersSlot, port: u16) {
    std::thread::spawn(move || match TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => {
            eprintln!("OpenAI-compatible API on http://127.0.0.1:{port}/v1 (model `{MODEL_ID}`)");
            for stream in listener.incoming().flatten() {
                let eng = engine.clone();
                let enc = encoders.clone();
                std::thread::spawn(move || {
                    let _ = handle(stream, eng, enc);
                });
            }
        }
        Err(e) => eprintln!("API server: could not bind port {port}: {e}"),
    });
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn handle(mut stream: TcpStream, engine: EngineSlot, encoders: EncodersSlot) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let t = h.trim_end();
        if t.is_empty() {
            break;
        }
        let lower = t.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }

    if method == "OPTIONS" {
        return write_head(&mut stream, "204 No Content", None, 0);
    }
    let p = path.trim_end_matches('/');
    match (method.as_str(), p) {
        ("GET", "/v1/models") | ("GET", "/models") => {
            let body = json!({
                "object": "list",
                "data": [{ "id": MODEL_ID, "object": "model", "created": now_secs(), "owned_by": "local" }]
            })
            .to_string();
            write_json(&mut stream, "200 OK", &body)
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            handle_chat(&mut stream, &body, &engine, &encoders)
        }
        ("GET", "") | ("GET", "/health") => write_json(&mut stream, "200 OK", "{\"status\":\"ok\"}"),
        _ => write_json(&mut stream, "404 Not Found", "{\"error\":{\"message\":\"not found\"}}"),
    }
}

/// Extract text from an OpenAI message content (string, or array of {type:text,text}).
fn content_str(c: &Value) -> String {
    if let Some(s) = c.as_str() {
        return s.to_string();
    }
    if let Some(arr) = c.as_array() {
        return arr
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
    }
    String::new()
}

/// Build the model prompt from OpenAI-format messages (incl. tool results + assistant tool-calls).
/// A trailing `assistant` message with plain text content is treated as a PREFILL (Anthropic-style):
/// the prompt ends with `ASSISTANT: <prefill>` and the model continues from it — used to force the
/// UI-grounding format by prefilling "(".
fn build_prompt(messages: &[Value]) -> String {
    let prefill = messages
        .last()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .filter(|m| m.get("tool_calls").is_none())
        .map(|m| content_str(m.get("content").unwrap_or(&Value::Null)))
        .filter(|c| !c.trim().is_empty());
    let body = if prefill.is_some() { &messages[..messages.len() - 1] } else { messages };

    let mut s = String::new();
    for m in body {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = content_str(m.get("content").unwrap_or(&Value::Null));
        match role {
            "system" => s.push_str(&format!("SYSTEM: {content}\n")),
            "tool" => {
                // The model was trained on JSON tool results — a plain-text result makes it re-call
                // instead of summarizing. Wrap non-JSON content as {"output": ...}.
                let body = if serde_json::from_str::<Value>(content.trim()).is_ok() {
                    content.clone()
                } else {
                    json!({ "output": content }).to_string()
                };
                s.push_str(&format!("TOOL: {body}\n"));
            }
            "assistant" => {
                if let Some(tcs) = m.get("tool_calls").and_then(|t| t.as_array()) {
                    // re-render the call(s) the model made: [{"name":..,"arguments":{..}}]
                    let calls: Vec<Value> = tcs
                        .iter()
                        .filter_map(|tc| {
                            let f = tc.get("function")?;
                            let name = f.get("name")?.as_str()?;
                            let args: Value = f
                                .get("arguments")
                                .and_then(|a| a.as_str())
                                .and_then(|t| serde_json::from_str(t).ok())
                                .unwrap_or_else(|| json!({}));
                            Some(json!({ "name": name, "arguments": args }))
                        })
                        .collect();
                    s.push_str(&format!("ASSISTANT: {}\n", Value::Array(calls)));
                } else {
                    s.push_str(&format!("ASSISTANT: {content}\n"));
                }
            }
            _ => s.push_str(&format!("USER: {content}\n")),
        }
    }
    s.push_str("ASSISTANT:");
    if let Some(pf) = prefill {
        s.push(' ');
        s.push_str(pf.trim_end());
    }
    s
}

/// Decode an OpenAI `image_url.url`: data-URI (base64), http(s) (fetched via ureq), or a local path.
fn fetch_image(url: &str) -> Option<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let comma = rest.find(',')?;
        let (meta, data) = (&rest[..comma], &rest[comma + 1..]);
        if meta.contains("base64") {
            base64::engine::general_purpose::STANDARD.decode(data.trim()).ok()
        } else {
            Some(data.as_bytes().to_vec())
        }
    } else if url.starts_with("http://") || url.starts_with("https://") {
        let resp = ureq::get(url).call().ok()?;
        let mut buf = Vec::new();
        resp.into_reader().read_to_end(&mut buf).ok()?;
        Some(buf)
    } else {
        std::fs::read(url).ok()
    }
}

/// Pull every `image_url` out of the OpenAI messages (content arrays).
fn collect_images(messages: &[Value]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for m in messages {
        if let Some(arr) = m.get("content").and_then(|c| c.as_array()) {
            for part in arr {
                if part.get("type").and_then(|t| t.as_str()) == Some("image_url") {
                    if let Some(url) = part.get("image_url").and_then(|i| i.get("url")).and_then(|u| u.as_str()) {
                        if let Some(bytes) = fetch_image(url) {
                            out.push(bytes);
                        }
                    }
                }
            }
        }
    }
    out
}

fn handle_chat(stream: &mut TcpStream, body: &[u8], engine: &EngineSlot, encoders: &EncodersSlot) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return write_json(stream, "400 Bad Request", &err_json(&format!("invalid JSON: {e}"))),
    };
    let eng = match engine.lock().unwrap().clone() {
        Some(e) => e,
        None => {
            return write_json(
                stream,
                "503 Service Unavailable",
                &err_json("no model loaded — load a GGUF in the hobby-chat window first"),
            )
        }
    };

    let messages = req.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    let prompt = build_prompt(&messages);

    // Encode any images (data-URI / URL) via the local SigLIP2 encoder; splice as prefix embeddings.
    let mut embeds: Vec<Vec<f32>> = Vec::new();
    for img in collect_images(&messages) {
        let rows = {
            let mut enc = encoders.lock().unwrap();
            enc.encode_image_bytes(&img)
        };
        match rows {
            Ok(r) => embeds.extend(r),
            Err(e) => return write_json(stream, "400 Bad Request", &err_json(&format!("image encode failed: {e}"))),
        }
    }
    let temp = req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32;
    let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512).clamp(1, 4096) as usize;
    let streaming = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let model = req.get("model").and_then(|v| v.as_str()).unwrap_or(MODEL_ID).to_string();
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let seed = req.get("seed").and_then(|v| v.as_u64()).unwrap_or(seq ^ now_secs());
    let opts = GenOpts { max_new: max_tokens, temp, top_p: 0.95, seed, rep_penalty: 1.3 };
    let id = format!("chatcmpl-{seq}");
    let created = now_secs();

    // ---- OpenAI function calling ----
    // If `tools` are present, force a call on the first turn (the 500M over-abstains otherwise) and
    // return OpenAI `tool_calls`; once a tool result is in the messages, answer in prose instead.
    let tool_choice = req.get("tool_choice").and_then(|c| c.as_str()).unwrap_or("auto");
    if let Some(tarr) = req.get("tools").and_then(|t| t.as_array()).filter(|a| !a.is_empty()) {
        if tool_choice != "none" {
            let has_tool_result =
                messages.iter().any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
            // Seed "[" only to overcome the 500M's initial over-abstention; on later turns generate
            // FREELY so the model can STOP (emit prose) when the task is done. parse_calls runs every
            // turn, so sequential compound tasks (one call per turn) still produce tool_calls until
            // the model finishes. The client loops while finish_reason == "tool_calls".
            let seed_bracket = tool_choice == "required" || !has_tool_result;
            let fns: Vec<Value> = tarr.iter().filter_map(|t| t.get("function").cloned()).collect();
            let schema = Value::Array(fns).to_string();
            let tprompt = format!("TOOLS: {schema}\n{prompt}");
            let fopts = GenOpts { max_new: 160, temp: 0.0, top_p: 1.0, seed, rep_penalty: 1.0 };
            let mut call = if seed_bracket { String::from("[") } else { String::new() };
            let genp = if seed_bracket { format!("{tprompt} [") } else { tprompt };
            eng.generate(&genp, &embeds, &fopts, |p| {
                call.push_str(p);
                true
            });
            let calls = crate::tools::parse_calls(&call);
            if !calls.is_empty() {
                let tool_calls: Vec<Value> = calls
                    .iter()
                    .enumerate()
                    .map(|(j, (name, args))| json!({
                        "id": format!("call_{seq}_{j}"),
                        "type": "function",
                        "function": { "name": name, "arguments": args.to_string() }
                    }))
                    .collect();
                let resp = json!({
                    "id": id, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "message": { "role": "assistant", "content": Value::Null, "tool_calls": Value::Array(tool_calls) }, "finish_reason": "tool_calls" }],
                    "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
                })
                .to_string();
                return write_json(stream, "200 OK", &resp);
            }
            // no parseable call -> the model answered in prose (it's done)
            let text = call.trim_start_matches('[').trim().to_string();
            let resp = json!({
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
            })
            .to_string();
            return write_json(stream, "200 OK", &resp);
        }
    }

    // When tools are present (e.g. the summary turn after a tool result), keep the TOOLS schema in the
    // prompt so the model stays in the trained format and summarizes instead of re-calling.
    let gen_prompt = match req.get("tools").and_then(|t| t.as_array()).filter(|a| !a.is_empty()) {
        Some(tarr) => {
            let fns: Vec<Value> = tarr.iter().filter_map(|t| t.get("function").cloned()).collect();
            format!("TOOLS: {}\n{prompt}", Value::Array(fns))
        }
        None => prompt.clone(),
    };

    if streaming {
        write_sse_head(stream)?;
        send_chunk(stream, &id, created, &model, Some("assistant"), None, None)?;
        let mut send_err = false;
        eng.generate(&gen_prompt, &embeds, &opts, |piece| {
            if !piece.is_empty() && send_chunk(stream, &id, created, &model, None, Some(piece), None).is_err() {
                send_err = true;
                return false; // client disconnected — stop generating
            }
            !send_err
        });
        send_chunk(stream, &id, created, &model, None, None, Some("stop"))?;
        stream.write_all(b"data: [DONE]\n\n")?;
        stream.flush()
    } else {
        let mut text = String::new();
        let n = eng.generate(&gen_prompt, &embeds, &opts, |piece| {
            text.push_str(piece);
            true
        });
        let resp = json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 0, "completion_tokens": n, "total_tokens": n }
        })
        .to_string();
        write_json(stream, "200 OK", &resp)
    }
}

fn send_chunk(
    stream: &mut TcpStream,
    id: &str,
    created: u64,
    model: &str,
    role: Option<&str>,
    content: Option<&str>,
    finish: Option<&str>,
) -> std::io::Result<()> {
    let mut delta = Map::new();
    if let Some(r) = role {
        delta.insert("role".into(), Value::String(r.into()));
    }
    if let Some(c) = content {
        delta.insert("content".into(), Value::String(c.into()));
    }
    let chunk = json!({
        "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{ "index": 0, "delta": Value::Object(delta), "finish_reason": finish }]
    });
    stream.write_all(format!("data: {chunk}\n\n").as_bytes())?;
    stream.flush()
}

fn err_json(msg: &str) -> String {
    json!({ "error": { "message": msg, "type": "invalid_request_error" } }).to_string()
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: *\r\n";

fn write_json(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{CORS}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn write_head(stream: &mut TcpStream, status: &str, _ct: Option<&str>, len: usize) -> std::io::Result<()> {
    let resp = format!("HTTP/1.1 {status}\r\n{CORS}Content-Length: {len}\r\nConnection: close\r\n\r\n");
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}

fn write_sse_head(stream: &mut TcpStream) -> std::io::Result<()> {
    let resp = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n");
    stream.write_all(resp.as_bytes())?;
    stream.flush()
}
