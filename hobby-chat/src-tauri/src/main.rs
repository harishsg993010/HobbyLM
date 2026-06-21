//! hobby-chat — Tauri backend. Embeds the hobby-rs Engine, exposes load_model + chat commands,
//! streams tokens to the webview, and runs the function-calling agent loop.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod computer;
mod encoders;
mod hf;
mod mcp;
mod server;
mod tools;

use base64::Engine as _;
use hobby_rs::imagegen::{ImageEngine, NEG_DEFAULT};
use hobby_rs::{Engine, GenOpts};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

struct AppState {
    engine: Arc<Mutex<Option<Arc<Engine>>>>,
    cancel: Arc<AtomicBool>,
    encoders: Arc<Mutex<encoders::Encoders>>,
    mcp: Arc<Mutex<mcp::McpManager>>,
    image: Arc<Mutex<Option<Arc<ImageEngine>>>>,
}

#[derive(Serialize, Clone)]
struct ModelInfo {
    arch: String,
    d_model: usize,
    n_layers: usize,
    n_experts: usize,
    top_k: usize,
    vocab: usize,
    ctx: usize,
    weights_gb: f64,
    has_speech: bool,
}

#[derive(Deserialize, Clone)]
struct Msg {
    role: String,
    content: String,
}

#[tauri::command]
fn load_model(path: String, quant: bool, state: State<AppState>) -> Result<ModelInfo, String> {
    let eng = Engine::load(std::path::Path::new(&path), quant).map_err(|e| e.to_string())?;

    let c = &eng.cfg;
    let info = ModelInfo {
        arch: c.arch.clone(),
        d_model: c.d_model,
        n_layers: c.n_layers,
        n_experts: c.n_experts,
        top_k: c.top_k,
        vocab: c.vocab_size,
        ctx: c.context_length,
        weights_gb: eng.weight_bytes() as f64 / 1e9,
        // image + speech encoders are always available (downloaded on first use via candle)
        has_speech: true,
    };
    *state.engine.lock().unwrap() = Some(Arc::new(eng));
    Ok(info)
}

/// Read a raw f32 embeddings file (N*d) into N rows of length d (e.g. image_embeds.bin / speech_embeds.bin).
fn read_embeds(path: &str, d: usize) -> Vec<Vec<f32>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    if d == 0 || bytes.is_empty() || bytes.len() % (d * 4) != 0 {
        return Vec::new();
    }
    let n = bytes.len() / (d * 4);
    (0..n)
        .map(|i| {
            (0..d)
                .map(|j| {
                    let o = (i * d + j) * 4;
                    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
                })
                .collect()
        })
        .collect()
}

/// Build "USER: ..\nASSISTANT: ..\n..." from the conversation (no trailing ASSISTANT:).
fn build_convo(messages: &[Msg]) -> String {
    let mut s = String::new();
    for m in messages {
        if m.role == "user" {
            s.push_str(&format!("USER: {}\n", m.content));
        } else {
            s.push_str(&format!("ASSISTANT: {}\n", m.content));
        }
    }
    s
}

fn run_chat(app: &AppHandle, eng: &Engine, messages: &[Msg], tools_on: bool, temp: f32, seed: u64, embeds: &[Vec<f32>], mcp: &Mutex<mcp::McpManager>, cancel: &AtomicBool) {
    let opts = GenOpts { max_new: 512, temp, top_p: 0.95, seed, rep_penalty: 1.3 };
    let convo = build_convo(messages);
    let cont = || !cancel.load(Ordering::Relaxed);

    if tools_on {
        let mcp_tools = mcp.lock().unwrap().enabled_tools();
        // Force a tool call (the model otherwise over-abstains): seed the assistant with " [".
        let prompt1 = format!("TOOLS: {}\n{}ASSISTANT:", tools::schema_json(&mcp_tools), convo);
        let mut call = String::from("[");
        let force = GenOpts { max_new: 128, temp: 0.0, top_p: 1.0, seed, rep_penalty: 1.0 };
        eng.generate(&format!("{prompt1} ["), &[], &force, |p| {
            call.push_str(p);
            cont()
        });
        if cancel.load(Ordering::Relaxed) {
            return;
        }

        if let Some((name, args)) = tools::parse_call(&call) {
            let _ = app.emit("tool_call", serde_json::json!({ "name": name, "args": args }));
            let result = if tools::is_builtin(&name) {
                tools::execute(&name, &args)
            } else {
                mcp.lock().unwrap().call(&name, &args).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
            };
            let _ = app.emit("tool_result", serde_json::json!({ "name": name, "result": result }));
            // feed the call + result back and stream the summary. The model was trained on JSON tool
            // results — wrap plain-text (e.g. MCP) results so it summarizes instead of re-calling.
            let tool_body = if serde_json::from_str::<serde_json::Value>(result.trim()).is_ok() {
                result.clone()
            } else {
                serde_json::json!({ "output": result }).to_string()
            };
            let prompt2 = format!("{prompt1} {}\nTOOL: {tool_body}\nASSISTANT:", call.trim());
            eng.generate(&prompt2, &[], &opts, |p| {
                let _ = app.emit("token", p);
                cont()
            });
        } else {
            let _ = app.emit("token", call.trim_start_matches('[').trim());
        }
    } else {
        let prompt = format!("{convo}ASSISTANT:");
        eng.generate(&prompt, embeds, &opts, |p| {
            let _ = app.emit("token", p);
            cont()
        });
    }
}

#[tauri::command]
fn chat(
    app: AppHandle,
    messages: Vec<Msg>,
    tools_on: bool,
    temp: f32,
    seed: u64,
    attachment: Option<String>,
    state: State<AppState>,
) -> Result<(), String> {
    let eng = state
        .engine
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "model not loaded".to_string())?;
    let embeds = match attachment {
        Some(p) => {
            let e = read_embeds(&p, eng.cfg.d_model);
            if e.is_empty() {
                return Err(format!("could not read embeddings from {p} (expected raw f32, N×{})", eng.cfg.d_model));
            }
            e
        }
        None => Vec::new(),
    };
    state.cancel.store(false, Ordering::Relaxed);
    let cancel = state.cancel.clone();
    let mcp = state.mcp.clone();
    std::thread::spawn(move || {
        run_chat(&app, &eng, &messages, tools_on, temp, seed, &embeds, &mcp, &cancel);
        let _ = app.emit("done", ());
    });
    Ok(())
}

#[tauri::command]
fn stop(state: State<AppState>) {
    state.cancel.store(true, Ordering::Relaxed);
}

/// Write N×D embeddings to a temp .bin and return its path (reuses the attachment-splice plumbing).
fn embeds_to_tempfile(rows: &[Vec<f32>], tag: &str) -> Result<String, String> {
    let d = rows.first().map(|r| r.len()).unwrap_or(0);
    if d == 0 {
        return Err("encoder produced no embeddings".into());
    }
    let mut bytes = Vec::with_capacity(rows.len() * d * 4);
    for r in rows {
        for &x in r {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    let path = std::env::temp_dir().join(format!("moe_{}_{}x{}.bin", tag, rows.len(), d));
    std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Encode mic PCM (mono f32 @ 16 kHz) into speech embeddings via the local Whisper encoder.
#[tauri::command]
fn encode_speech(pcm: Vec<f32>, state: State<AppState>) -> Result<String, String> {
    let rows = {
        let mut enc = state.encoders.lock().unwrap();
        enc.encode_speech(&pcm).map_err(|e| e.to_string())?
    };
    embeds_to_tempfile(&rows, "voice")
}

/// Encode an image file (PNG/JPG/…) into vision embeddings via the local SigLIP2 encoder.
#[tauri::command]
fn encode_image(path: String, state: State<AppState>) -> Result<String, String> {
    let rows = {
        let mut enc = state.encoders.lock().unwrap();
        enc.encode_image(&path).map_err(|e| e.to_string())?
    };
    embeds_to_tempfile(&rows, "image")
}

// ---- text-to-image (HobbyLM-Image: CLIP + DiT + DC-AE, all in hobby-rs) ----

#[derive(Serialize, Clone)]
struct ImageModelInfo {
    resolution: usize,
}

/// Load the exported image weights directory (dit/clip/dcae safetensors + metas). Returns the
/// output resolution. Heavy (mmaps ~2.5 GB); call once.
#[tauri::command]
fn load_image_model(dir: String, state: State<AppState>) -> Result<ImageModelInfo, String> {
    let eng = ImageEngine::load(std::path::Path::new(&dir)).map_err(|e| e.to_string())?;
    let info = ImageModelInfo { resolution: eng.resolution() };
    *state.image.lock().unwrap() = Some(Arc::new(eng));
    Ok(info)
}

/// Generate an image from `prompt` on a background thread. Emits `image_progress` {step,total} per
/// sampler step and `image_done` { data_uri } (a base64 PNG) when finished, or `image_error`.
#[tauri::command]
fn generate_image(
    app: AppHandle,
    prompt: String,
    neg: Option<String>,
    steps: usize,
    cfg: f32,
    seed: u64,
    state: State<AppState>,
) -> Result<(), String> {
    let eng = state
        .image
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "image model not loaded".to_string())?;
    let neg = neg.unwrap_or_else(|| NEG_DEFAULT.to_string());
    let steps = steps.clamp(1, 250);
    std::thread::spawn(move || {
        let app2 = app.clone();
        let (rgb, w, h) = eng.generate(&prompt, &neg, steps, cfg, seed, |s, tot| {
            let _ = app2.emit("image_progress", serde_json::json!({ "step": s, "total": tot }));
        });
        let png = hobby_rs::png::encode_rgb(&rgb, w, h);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let _ = app.emit(
            "image_done",
            serde_json::json!({ "data_uri": format!("data:image/png;base64,{b64}"), "w": w, "h": h }),
        );
    });
    Ok(())
}

// ---- MCP ----

/// Connect (or reconnect) a stdio MCP server. Runs on a fresh std::thread so tokio's block_on inside
/// the manager never executes on a tokio worker thread. Returns the server's tool count.
#[tauri::command]
fn mcp_add(name: String, command: String, args: Vec<String>, state: State<AppState>) -> Result<usize, String> {
    let mcp = state.mcp.clone();
    std::thread::spawn(move || mcp.lock().unwrap().add(&name, &command, &args).map_err(|e| e.to_string()))
        .join()
        .map_err(|_| "mcp_add thread panicked".to_string())?
}

#[tauri::command]
fn mcp_remove(name: String, state: State<AppState>) {
    let mcp = state.mcp.clone();
    let _ = std::thread::spawn(move || mcp.lock().unwrap().remove(&name)).join();
}

/// Current servers + their tools (each with an `enabled` flag).
#[tauri::command]
fn mcp_status(state: State<AppState>) -> Vec<serde_json::Value> {
    state.mcp.lock().unwrap().server_summaries()
}

/// Enable/disable a single tool (only enabled tools are exposed to the model).
#[tauri::command]
fn mcp_set_tool(server: String, tool: String, enabled: bool, state: State<AppState>) {
    state.mcp.lock().unwrap().set_tool_enabled(&server, &tool, enabled);
}

/// List visible top-level windows for the computer-use target picker.
#[tauri::command]
fn computer_windows() -> Result<Vec<String>, String> {
    computer::list_windows()
}

/// Model-driven computer use: snapshot `window`'s accessibility tree, let the model pick one grounded
/// action, then preview (execute=false) or perform it (execute=true).
#[tauri::command]
fn computer_use(
    instruction: String,
    window: String,
    execute: bool,
    state: State<AppState>,
) -> Result<serde_json::Value, String> {
    let eng = state
        .engine
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "no model loaded".to_string())?;
    computer::run(eng, &instruction, &window, execute)
}

/// Model-driven PLANNING: snapshot->plan->ground->execute loop for a high-level `goal` (v4), until the
/// model emits `finish` or `max_steps`. Returns the full step trajectory.
#[tauri::command]
fn computer_goal(
    goal: String,
    window: String,
    max_steps: usize,
    execute: bool,
    state: State<AppState>,
) -> Result<serde_json::Value, String> {
    let eng = state
        .engine
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "no model loaded".to_string())?;
    computer::run_goal(eng, &goal, &window, max_steps.clamp(1, 15), execute)
}

/// Minimal WAV reader (16-bit PCM) -> mono f32 @ 16 kHz, for the speech smoke test. Finds the `data`
/// chunk, reads i16 samples, averages channels. Assumes the file is already 16 kHz.
fn read_wav_16k_mono(path: &str) -> Option<Vec<f32>> {
    let b = std::fs::read(path).ok()?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let mut channels = 1usize;
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = i + 8;
        if id == b"fmt " && body + 16 <= b.len() {
            channels = u16::from_le_bytes([b[body + 2], b[body + 3]]).max(1) as usize;
        } else if id == b"data" {
            let end = (body + sz).min(b.len());
            let samples: Vec<f32> = b[body..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
            if channels <= 1 {
                return Some(samples);
            }
            return Some(samples.chunks(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect());
        }
        i = body + sz + (sz & 1);
    }
    None
}

fn main() {
    // Headless smoke tests:
    //   hobby-chat --test-image <path.png>     (SigLIP2 -> vision embeds)
    //   hobby-chat --test-speech               (synthetic tone -> Whisper speech embeds)
    let argv: Vec<String> = std::env::args().collect();
    // Headless computer-use test (snapshot a window, ask the model, preview or execute one action):
    //   HOBBYLM_CHAT_MODEL=...gguf  hobby-chat --test-computer "click equals" "Calculator" [--execute]
    if argv.len() >= 2 && argv[1] == "--test-computer" {
        let path = std::env::var("HOBBYLM_CHAT_MODEL").expect("set HOBBYLM_CHAT_MODEL=<gguf>");
        let instr = argv.get(2).map(|s| s.as_str()).unwrap_or("click the equals button");
        let window = argv.get(3).filter(|s| !s.starts_with("--")).map(|s| s.as_str()).unwrap_or("Calculator");
        let execute = argv.iter().any(|a| a == "--execute");
        eprintln!("[test-computer] loading {path} …");
        let eng = Arc::new(Engine::load(std::path::Path::new(&path), true).expect("load model"));
        match computer::run(eng, instr, window, execute) {
            Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
            Err(e) => {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    // Headless planning test (decompose a goal + run the loop on a window):
    //   HOBBYLM_CHAT_MODEL=...gguf  hobby-chat --test-goal "calculate 7 plus 2" "Calculator" [--execute]
    if argv.len() >= 2 && argv[1] == "--test-goal" {
        let path = std::env::var("HOBBYLM_CHAT_MODEL").expect("set HOBBYLM_CHAT_MODEL=<gguf>");
        let goal = argv.get(2).map(|s| s.as_str()).unwrap_or("calculate 7 plus 2");
        let window = argv.get(3).filter(|s| !s.starts_with("--")).map(|s| s.as_str()).unwrap_or("Calculator");
        let execute = argv.iter().any(|a| a == "--execute");
        eprintln!("[test-goal] loading {path} …");
        let eng = Arc::new(Engine::load(std::path::Path::new(&path), true).expect("load model"));
        match computer::run_goal(eng, goal, window, 12, execute) {
            Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
            Err(e) => {
                eprintln!("ERROR: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    if argv.len() >= 2 && (argv[1] == "--test-speech" || argv[1] == "--test-image") {
        let mut enc = encoders::Encoders::new();
        let rows = if argv[1] == "--test-image" {
            let path = argv.get(2).expect("usage: --test-image <path>");
            eprintln!("[test] loading SigLIP2 + encoding {path} …");
            enc.encode_image(path).expect("encode image")
        } else if let Some(wav) = argv.get(2) {
            eprintln!("[test] loading Whisper + encoding {wav} …");
            let pcm = read_wav_16k_mono(wav).expect("read wav");
            enc.encode_speech(&pcm).expect("encode speech")
        } else {
            eprintln!("[test] loading Whisper + encoding synthetic tone …");
            let pcm: Vec<f32> = (0..16000 * 3).map(|i| (i as f32 * 0.02).sin() * 0.1).collect();
            enc.encode_speech(&pcm).expect("encode speech")
        };
        let out = embeds_to_tempfile(&rows, "test").expect("write");
        println!("OK embeds: {} rows x {} -> {out}", rows.len(), rows.first().map(|r| r.len()).unwrap_or(0));
        println!("first row[..5] = {:?}", &rows[0][..5.min(rows[0].len())]);
        return;
    }

    // Full LLM->MCP loop: `hobby-chat --test-mcp-llm <gguf> [fs|everything]`
    if argv.len() >= 3 && argv[1] == "--test-mcp-llm" {
        let eng = Engine::load(std::path::Path::new(&argv[2]), true).expect("load gguf");
        let mut mgr = mcp::McpManager::new().expect("mcp runtime");
        let mode = argv.get(3).map(|s| s.as_str()).unwrap_or("everything");
        let dir = "C:/Users/haris/Desktop/personal/LLM_training/moe-lab";
        let (args, prompts): (Vec<String>, Vec<String>) = if mode == "fs" {
            eprintln!("[test] connecting server-filesystem ({dir}) …");
            (
                vec!["-y".into(), "@modelcontextprotocol/server-filesystem".into(), dir.into()],
                vec![
                    format!("List the files in the directory {dir}"),
                    format!("Get information about the file {dir}/jfk.wav"),
                    format!("Read the text file {dir}/multimodal.py"),
                ],
            )
        } else {
            eprintln!("[test] connecting server-everything …");
            (
                vec!["-y".into(), "@modelcontextprotocol/server-everything".into()],
                vec![
                    "Use the echo tool to repeat the message: hello from the model.".into(),
                    "Add the numbers 7 and 5 using a tool.".into(),
                ],
            )
        };
        mgr.add("srv", "npx", &args).expect("connect");
        let mut mcp_tools = mgr.all_tools();
        // The 500M model degrades badly past a few tools — cap the exposed set to keep it tractable.
        if mode == "fs" {
            let keep = ["list_directory", "read_text_file", "get_file_info"];
            mcp_tools.retain(|t| keep.contains(&t.name.as_str()));
        }
        let schema = tools::schema_json(&mcp_tools);
        eprintln!("[test] {} tools in schema ({} chars): {:?}", mcp_tools.len() + 2, schema.len(),
            mcp_tools.iter().map(|t| t.name.clone()).collect::<Vec<_>>());

        for user in prompts.iter() {
            let prompt1 = format!("TOOLS: {schema}\nUSER: {user}\nASSISTANT:");
            let mut call = String::from("[");
            let force = GenOpts { max_new: 96, temp: 0.0, top_p: 1.0, seed: 1, rep_penalty: 1.0 };
            eng.generate(&format!("{prompt1} ["), &[], &force, |p| {
                call.push_str(p);
                true
            });
            println!("\nUSER: {user}\n  RAW: {}", call.trim());
            match tools::parse_call(&call) {
                Some((name, args)) => {
                    let builtin = tools::is_builtin(&name);
                    print!("  PARSED name={name} (builtin={builtin}) args={args}\n");
                    let result = if builtin {
                        tools::execute(&name, &args)
                    } else {
                        mgr.call(&name, &args).unwrap_or_else(|e| format!("<mcp error: {e}>"))
                    };
                    println!("  -> {} RESULT: {result}", if builtin { "BUILTIN" } else { "MCP" });
                }
                None => println!("  (no parseable tool call)"),
            }
        }
        return;
    }

    if argv.len() >= 2 && argv[1] == "--test-mcp" {
        let mut mgr = mcp::McpManager::new().expect("mcp runtime");
        eprintln!("[test] connecting @modelcontextprotocol/server-everything via npx …");
        let n = mgr
            .add("everything", "npx", &["-y".into(), "@modelcontextprotocol/server-everything".into()])
            .expect("connect");
        println!("connected: {n} tools: {:?}", mgr.all_tools().iter().map(|t| t.name.clone()).collect::<Vec<_>>());
        let res = mgr.call("echo", &serde_json::json!({ "message": "hello from hobby-chat" })).expect("call echo");
        println!("echo -> {res}");
        let add = mgr.call("add", &serde_json::json!({ "a": 7, "b": 5 })).expect("call add");
        println!("add(7,5) -> {add}");
        return;
    }

    let mcp = Arc::new(Mutex::new(mcp::McpManager::new().expect("init MCP runtime")));
    load_mcp_config(&mcp); // connect any servers listed in mcp.json

    // OpenAI-compatible HTTP API backed by the same Engine the UI uses.
    let engine: Arc<Mutex<Option<Arc<Engine>>>> = Arc::new(Mutex::new(None));
    // Optional: auto-load a model at startup (so the API works without opening the window).
    if let Ok(path) = std::env::var("HOBBYLM_CHAT_MODEL") {
        match Engine::load(std::path::Path::new(&path), true) {
            Ok(e) => {
                eprintln!("auto-loaded model: {path}");
                *engine.lock().unwrap() = Some(Arc::new(e));
            }
            Err(e) => eprintln!("HOBBYLM_CHAT_MODEL load failed: {e}"),
        }
    }
    let encoders = Arc::new(Mutex::new(encoders::Encoders::new()));
    server::spawn(engine.clone(), encoders.clone(), 11250);

    // Optional: auto-load the image model at startup from HOBBYLM_IMAGE_DIR.
    let image: Arc<Mutex<Option<Arc<ImageEngine>>>> = Arc::new(Mutex::new(None));
    if let Ok(dir) = std::env::var("HOBBYLM_IMAGE_DIR") {
        match ImageEngine::load(std::path::Path::new(&dir)) {
            Ok(e) => {
                eprintln!("auto-loaded image model ({}px) from {dir}", e.resolution());
                *image.lock().unwrap() = Some(Arc::new(e));
            }
            Err(e) => eprintln!("HOBBYLM_IMAGE_DIR load failed: {e}"),
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            engine,
            cancel: Arc::new(AtomicBool::new(false)),
            encoders,
            mcp,
            image,
        })
        .invoke_handler(tauri::generate_handler![
            load_model, chat, stop, encode_speech, encode_image, mcp_add, mcp_remove, mcp_status,
            mcp_set_tool, computer_windows, computer_use, computer_goal, load_image_model,
            generate_image
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Connect MCP servers declared in `mcp.json` next to the exe (Claude-Desktop-style
/// `{"mcpServers":{"name":{"command":..,"args":[..]}}}`). Best-effort; logs failures.
fn load_mcp_config(mcp: &Arc<Mutex<mcp::McpManager>>) {
    let path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("mcp.json")));
    let Some(path) = path else { return };
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) else {
        eprintln!("mcp.json: invalid JSON");
        return;
    };
    if let Some(servers) = cfg.get("mcpServers").and_then(|v| v.as_object()) {
        for (name, spec) in servers {
            let command = spec.get("command").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let args: Vec<String> = spec
                .get("args")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if command.is_empty() {
                continue;
            }
            match mcp.lock().unwrap().add(name, &command, &args) {
                Ok(n) => eprintln!("MCP `{name}`: connected, {n} tools"),
                Err(e) => eprintln!("MCP `{name}`: {e}"),
            }
        }
    }
}
