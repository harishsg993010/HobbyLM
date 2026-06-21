const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const sendBtn = $("send");
const toolsToggle = $("toolsToggle");

let history = []; // [{role, content}]
let loaded = false;
let generating = false;
let thread = null;
let cur = null; // { el, content, text } of the streaming assistant message
let curToolCard = null;
let attachment = null; // { path, name } of a precomputed embeddings .bin

// ---------- model load ----------
$("loadBtn").onclick = loadModel;
$("modelPath").addEventListener("keydown", (e) => { if (e.key === "Enter") loadModel(); });

$("browseBtn").onclick = async () => {
  try {
    const sel = await invoke("plugin:dialog|open", {
      options: {
        multiple: false,
        directory: false,
        title: "Select a GGUF model",
        filters: [{ name: "GGUF model", extensions: ["gguf"] }],
      },
    });
    if (sel) {
      const path = typeof sel === "string" ? sel : sel.path ?? String(sel);
      $("modelPath").value = path;
      loadModel();
    }
  } catch (e) {
    const st = $("loadStatus");
    st.className = "status error";
    st.textContent = "file picker error: " + e;
  }
};

async function loadModel() {
  const path = $("modelPath").value.trim();
  if (!path) return;
  const st = $("loadStatus");
  st.className = "status";
  st.textContent = "loading… (quantizing to Q8, ~few seconds)";
  $("loadBtn").disabled = true;
  try {
    const info = await invoke("load_model", { path, quant: true });
    loaded = true;
    $("modelBadge").textContent =
      `${info.arch} · ${info.n_experts}/${info.top_k} experts · ${info.weights_gb.toFixed(2)} GB`;
    $("setup").style.display = "none";
    sendBtn.disabled = false;
    $("attachBtn").disabled = false;
    if (info.has_speech) {
      const mb = $("micBtn");
      mb.hidden = false;
      mb.disabled = false;
    }
    inputEl.focus();
  } catch (e) {
    st.className = "status error";
    st.textContent = "" + e;
    $("loadBtn").disabled = false;
  }
}

// ---------- attachment: image upload (SigLIP2) or mic (Whisper), both -> spliced embeddings ----------
const KIND_ICON = { image: "🖼", audio: "🎤" };

function renderChip() {
  const chip = $("attachChip");
  if (!attachment) { chip.style.display = "none"; return; }
  $("attachName").textContent = attachment.name;
  $("attachMeta").textContent = attachment.kind || "embeddings";
  chip.querySelector(".chip-icon").textContent = KIND_ICON[attachment.kind] || "📎";
  chip.style.display = "";
}

$("attachBtn").onclick = async () => {
  if (!loaded || generating) return;
  let sel;
  try {
    sel = await invoke("plugin:dialog|open", {
      options: {
        multiple: false,
        directory: false,
        title: "Upload an image",
        filters: [{ name: "Image", extensions: ["png", "jpg", "jpeg", "webp", "bmp", "gif"] }],
      },
    });
  } catch (e) {
    return; // cancelled / picker error
  }
  if (!sel) return;
  const path = typeof sel === "string" ? sel : sel.path ?? String(sel);
  const name = path.split(/[\\/]/).pop();

  $("attachBtn").disabled = true;
  micStatus("encoding image… (first use downloads the SigLIP2 encoder)");
  try {
    const binPath = await invoke("encode_image", { path });
    attachment = { path: binPath, name, kind: "image" };
    renderChip();
    micStatus("", false);
  } catch (e) {
    micStatus("image encode failed: " + e, true);
    setTimeout(() => micStatus("", false), 4000);
  }
  $("attachBtn").disabled = false;
};

$("attachRemove").onclick = () => { attachment = null; renderChip(); };

// ---------- microphone (live speech → Whisper encoder → MoE) ----------
let recording = false;
let micCtx = null, micStream = null, micNode = null, micChunks = [], micTimer = null;
const MIC_MAX_SEC = 30; // Whisper window

function micStatus(text, show = true) {
  const el = $("micStatus");
  el.textContent = text;
  el.style.display = show ? "" : "none";
}

async function startRecording() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    micStatus("microphone blocked: " + e, true);
    setTimeout(() => micStatus("", false), 3000);
    return;
  }
  micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const src = micCtx.createMediaStreamSource(micStream);
  micNode = micCtx.createScriptProcessor(4096, 1, 1);
  micChunks = [];
  micNode.onaudioprocess = (e) => {
    micChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  src.connect(micNode);
  micNode.connect(micCtx.destination); // processor emits silence — no feedback

  recording = true;
  const mb = $("micBtn");
  mb.classList.add("recording");
  mb.title = "Stop recording";
  const t0 = Date.now();
  micStatus("● recording… 0.0s");
  micTimer = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    micStatus("● recording… " + s.toFixed(1) + "s");
    if (s >= MIC_MAX_SEC) stopRecording();
  }, 100);
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  clearInterval(micTimer);
  const mb = $("micBtn");
  mb.classList.remove("recording");
  mb.title = "Record voice (speech → MoE)";
  mb.disabled = true;

  try {
    micNode.disconnect();
    micStream.getTracks().forEach((t) => t.stop());
  } catch (e) {}

  // flatten captured 16 kHz mono PCM
  const total = micChunks.reduce((a, c) => a + c.length, 0);
  const pcm = new Float32Array(Math.min(total, MIC_MAX_SEC * 16000));
  let o = 0;
  for (const c of micChunks) {
    if (o >= pcm.length) break;
    const take = Math.min(c.length, pcm.length - o);
    pcm.set(c.subarray(0, take), o);
    o += take;
  }
  try { await micCtx.close(); } catch (e) {}

  if (o < 1600) { // < 0.1 s — nothing useful
    micStatus("too short — hold longer", true);
    setTimeout(() => micStatus("", false), 2000);
    mb.disabled = false;
    return;
  }

  micStatus("encoding speech…");
  try {
    const path = await invoke("encode_speech", { pcm: Array.from(pcm) });
    attachment = { path, name: "voice (" + (o / 16000).toFixed(1) + "s)", kind: "audio" };
    renderChip();
    micStatus("", false);
  } catch (e) {
    micStatus("encode failed: " + e, true);
    setTimeout(() => micStatus("", false), 3000);
  }
  mb.disabled = false;
}

$("micBtn").onclick = () => { if (recording) stopRecording(); else startRecording(); };

// ---------- composer ----------
function autosize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + "px";
}
inputEl.addEventListener("input", autosize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!generating) send(); }
});

const ICON_SEND = `<svg viewBox="0 0 24 24" width="20" height="20"><path d="M12 21V6m0 0l-6 6m6-6l6 6"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>`;
const ICON_STOP = `<svg viewBox="0 0 24 24" width="18" height="18"><rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/></svg>`;

function setSendMode(streaming) {
  if (streaming) {
    sendBtn.classList.add("stop");
    sendBtn.innerHTML = ICON_STOP;
    sendBtn.title = "Stop";
    sendBtn.disabled = false;
  } else {
    sendBtn.classList.remove("stop");
    sendBtn.innerHTML = ICON_SEND;
    sendBtn.title = "Send";
    sendBtn.disabled = !loaded;
  }
}

async function stopGen() {
  sendBtn.disabled = true; // visual; finish() re-enables
  try { await invoke("stop"); } catch (e) {}
}

sendBtn.onclick = () => { if (generating) stopGen(); else send(); };

$("newChat").onclick = () => {
  if (generating) return;
  history = [];
  if (thread) thread.remove();
  thread = null;
  attachment = null;
  renderChip();
  $("empty").style.display = "";
};

function ensureThread() {
  $("empty").style.display = "none";
  if (!thread) {
    thread = document.createElement("div");
    thread.className = "thread";
    messagesEl.appendChild(thread);
  }
}

function addMessage(role, text) {
  ensureThread();
  const msg = document.createElement("div");
  msg.className = "msg " + role;
  msg.innerHTML =
    `<div class="avatar">${role === "assistant" ? "◆" : ""}</div>` +
    `<div class="body"><div class="content"></div></div>`;
  msg.querySelector(".content").textContent = text;
  thread.appendChild(msg);
  scrollDown();
  return msg;
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function send() {
  if (!loaded || generating) return;
  const text = inputEl.value.trim();
  if (!text && !attachment) return;

  const att = attachment; // applies only to this turn (embeds prefix the prompt)
  const umsg = addMessage("user", text);
  if (att) {
    const tag = document.createElement("div");
    tag.className = "msg-attach";
    tag.textContent = (KIND_ICON[att.kind] || "📎") + " " + att.name;
    umsg.querySelector(".body").insertBefore(tag, umsg.querySelector(".content"));
  }
  history.push({ role: "user", content: text });
  inputEl.value = "";
  autosize();
  attachment = null;
  renderChip();

  const msg = addMessage("assistant", "");
  const content = msg.querySelector(".content");
  content.classList.add("streaming");
  cur = { el: msg, content, text: "" };
  curToolCard = null;

  generating = true;
  setSendMode(true);

  invoke("chat", {
    messages: history,
    toolsOn: toolsToggle.checked,
    temp: 0.7,
    seed: Math.floor(Math.random() * 1e6),
    attachment: att ? att.path : null,
  }).catch((e) => {
    content.textContent += "\n[error] " + e;
    finish();
  });
}

function finish() {
  if (cur) {
    cur.content.classList.remove("streaming");
    history.push({ role: "assistant", content: cur.text });
  }
  cur = null;
  curToolCard = null;
  generating = false;
  setSendMode(false);
}

// ---------- streaming events ----------
listen("token", (e) => {
  if (!cur) return;
  cur.text += e.payload;
  cur.content.textContent = cur.text;
  scrollDown();
});

listen("tool_call", (e) => {
  if (!cur) return;
  const { name, args } = e.payload;
  const card = document.createElement("div");
  card.className = "tool-card";
  card.innerHTML =
    `<div class="tool-head">🔧 ${name}</div>` +
    `<div class="tool-body">${escapeHtml(JSON.stringify(args))}</div>` +
    `<div class="tool-result">running…</div>`;
  cur.content.parentElement.insertBefore(card, cur.content);
  curToolCard = card;
  scrollDown();
});

listen("tool_result", (e) => {
  if (curToolCard) {
    curToolCard.querySelector(".tool-result").textContent = "→ " + e.payload.result;
    scrollDown();
  }
});

listen("done", () => finish());

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// ---------- MCP servers ----------
const mcpModal = $("mcpModal");
$("mcpBtn").onclick = () => { mcpModal.style.display = "flex"; refreshMcp(); };
$("mcpClose").onclick = () => { mcpModal.style.display = "none"; };
mcpModal.addEventListener("click", (e) => { if (e.target === mcpModal) mcpModal.style.display = "none"; });

async function refreshMcp() {
  const list = $("mcpList");
  let servers = [];
  try { servers = await invoke("mcp_status"); } catch (e) {}
  if (!servers.length) {
    list.innerHTML = `<div class="mcp-empty">No servers connected yet.</div>`;
    return;
  }
  list.innerHTML = "";
  for (const s of servers) {
    const toolsArr = s.tools || [];
    const onCount = toolsArr.filter((t) => t.enabled).length;
    const el = document.createElement("div");
    el.className = "mcp-server";
    const tools = toolsArr
      .map(
        (t) =>
          `<label class="mcp-tool${t.enabled ? " on" : ""}">` +
          `<input type="checkbox" ${t.enabled ? "checked" : ""} data-tool="${escapeHtml(t.name)}">` +
          `${escapeHtml(t.name)}</label>`
      )
      .join("");
    el.innerHTML =
      `<div class="mcp-server-head">` +
      `<span class="mcp-server-name">${escapeHtml(s.server)}<span class="count">${onCount}/${toolsArr.length} on</span></span>` +
      `<button class="mcp-rm">Remove</button></div>` +
      `<div class="mcp-hint-row">Only checked tools are given to the model — keep it to a few (500M handles ~3 best).</div>` +
      `<div class="mcp-tools">${tools}</div>`;
    el.querySelector(".mcp-rm").onclick = async () => {
      try { await invoke("mcp_remove", { name: s.server }); } catch (e) {}
      refreshMcp();
    };
    el.querySelectorAll('.mcp-tool input[type="checkbox"]').forEach((cb) => {
      cb.onchange = async () => {
        try {
          await invoke("mcp_set_tool", { server: s.server, tool: cb.dataset.tool, enabled: cb.checked });
          cb.parentElement.classList.toggle("on", cb.checked);
          const head = el.querySelector(".count");
          const n = el.querySelectorAll('.mcp-tool input:checked').length;
          head.textContent = `${n}/${toolsArr.length} on`;
        } catch (e) {}
      };
    });
    list.appendChild(el);
  }
}

$("mcpConnect").onclick = async () => {
  const name = $("mcpName").value.trim();
  const command = $("mcpCommand").value.trim();
  const argsRaw = $("mcpArgs").value.trim();
  const st = $("mcpStatus");
  if (!name || !command) { st.className = "status error"; st.textContent = "name and command are required"; return; }
  const args = argsRaw.length ? argsRaw.split(/\s+/) : [];
  st.className = "status";
  st.textContent = "connecting…";
  $("mcpConnect").disabled = true;
  try {
    const n = await invoke("mcp_add", { name, command, args });
    st.textContent = `connected: ${n} tools`;
    $("mcpName").value = ""; $("mcpCommand").value = ""; $("mcpArgs").value = "";
    refreshMcp();
  } catch (e) {
    st.className = "status error";
    st.textContent = "" + e;
  }
  $("mcpConnect").disabled = false;
};

// ---------- computer use ----------
const computerModal = $("computerModal");
const cuLog = $("computerLog");

function cuLine(html, cls) {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.innerHTML = html;
  cuLog.appendChild(div);
  cuLog.scrollTop = cuLog.scrollHeight;
}

async function refreshWindows() {
  const sel = $("computerWindow");
  const prev = sel.value;
  try {
    const wins = await invoke("computer_windows");
    sel.innerHTML = "";
    wins.forEach((w) => {
      const o = document.createElement("option");
      o.value = w; o.textContent = w;
      sel.appendChild(o);
    });
    if (prev && wins.includes(prev)) sel.value = prev;
  } catch (e) { cuLine("window list failed: " + e, "cu-fail"); }
}

const DEFAULT_CU_MODEL = "computeruse-v3-hobbylm.gguf";

function showCuModelStatus() {
  const st = $("computerModelStatus");
  if (loaded) {
    st.className = "status";
    st.textContent = "✓ loaded: " + $("modelBadge").textContent;
  } else {
    st.className = "status error";
    st.textContent = "no model loaded — load a computer-use GGUF first";
  }
}

async function loadCuModel() {
  const path = $("computerModel").value.trim();
  const st = $("computerModelStatus");
  if (!path) { st.className = "status error"; st.textContent = "enter a GGUF path"; return; }
  st.className = "status";
  st.textContent = "loading… (quantizing to Q8, ~few seconds)";
  $("computerLoadBtn").disabled = true;
  try {
    const info = await invoke("load_model", { path, quant: true });
    loaded = true;
    $("modelBadge").textContent =
      `${info.arch} · ${info.n_experts}/${info.top_k} experts · ${info.weights_gb.toFixed(2)} GB`;
    $("setup").style.display = "none";
    sendBtn.disabled = false;
    $("attachBtn").disabled = false;
    showCuModelStatus();
  } catch (e) {
    st.className = "status error";
    st.textContent = "" + e;
  }
  $("computerLoadBtn").disabled = false;
}

$("computerBtn").onclick = () => {
  computerModal.style.display = "flex";
  if (!$("computerModel").value) $("computerModel").value = DEFAULT_CU_MODEL;
  showCuModelStatus();
  refreshWindows();
};
$("computerClose").onclick = () => { computerModal.style.display = "none"; };
computerModal.addEventListener("click", (e) => { if (e.target === computerModal) computerModal.style.display = "none"; });
$("computerRefresh").onclick = refreshWindows;
$("computerLoadBtn").onclick = loadCuModel;
$("computerModel").addEventListener("keydown", (e) => { if (e.key === "Enter") loadCuModel(); });
$("computerBrowse").onclick = async () => {
  try {
    const sel = await invoke("plugin:dialog|open", {
      options: { multiple: false, directory: false, title: "Select computer-use GGUF",
                 filters: [{ name: "GGUF model", extensions: ["gguf"] }] },
    });
    if (sel) $("computerModel").value = typeof sel === "string" ? sel : (sel.path ?? String(sel));
  } catch (e) {}
};

function esc(s) { return ("" + s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

function renderStep(s, execute) {
  const a = s.action;
  if (a && a.name) {
    const args = a.arguments ? " " + esc(JSON.stringify(a.arguments)) : "";
    cuLine(`<span class="cu-action">${s.step}. ${esc(a.name)}</span>${args}`);
  }
  const ok = !/^(UNRESOLVED|FAILED|UNSUPPORTED)/.test(s.result || "");
  cuLine("   " + (execute ? "▶ " : "👁 ") + esc(s.result || ""), ok ? "cu-ok" : "cu-fail");
}

async function runComputer(execute) {
  const window = $("computerWindow").value;
  const instruction = $("computerInstr").value.trim();
  const plan = $("computerPlan").checked;
  if (!window) { cuLine("pick a window first", "cu-fail"); return; }
  if (!instruction) { cuLine(plan ? "type a goal" : "type an instruction", "cu-fail"); return; }
  $("computerPreview").disabled = true; $("computerRun").disabled = true;
  try {
    if (plan) {
      cuLine(`<span class="cu-dim">⊕ GOAL: ${esc(instruction)}  [${esc(window)}]</span>`);
      const r = await invoke("computer_goal", { goal: instruction, window, maxSteps: 12, execute });
      (r.steps || []).forEach((s) => renderStep(s, execute));
      cuLine(`<span class="cu-dim">— ${(r.steps || []).length} steps —</span>`);
    } else {
      cuLine(`<span class="cu-dim">› ${esc(instruction)}  [${esc(window)}]</span>`);
      const r = await invoke("computer_use", { instruction, window, execute });
      const a = r.action;
      cuLine(`<span class="cu-action">${esc(a.name)}</span> ${esc(JSON.stringify(a.arguments))} ` +
             `<span class="cu-dim">(${r.n_elements} elems)</span>`);
      const ok = !/^(UNRESOLVED|FAILED|UNSUPPORTED)/.test(r.result);
      cuLine((execute ? "▶ " : "👁 ") + esc(r.result), ok ? "cu-ok" : "cu-fail");
    }
  } catch (e) {
    cuLine("error: " + esc(e), "cu-fail");
  }
  $("computerPreview").disabled = false; $("computerRun").disabled = false;
}

$("computerPreview").onclick = () => runComputer(false);
$("computerRun").onclick = () => runComputer(true);
$("computerInstr").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); runComputer(true); }
});

// ---- text-to-image (HobbyLM-Image) ----
const imageModal = $("imageModal");
let imageBusy = false;
$("imageBtn").onclick = () => { imageModal.style.display = "flex"; };
$("imageClose").onclick = () => { imageModal.style.display = "none"; };
imageModal.addEventListener("click", (e) => { if (e.target === imageModal) imageModal.style.display = "none"; });
$("imageBrowse").onclick = async () => {
  try {
    const sel = await invoke("plugin:dialog|open", {
      options: { multiple: false, directory: true, title: "Select exported image-weights directory" },
    });
    if (sel) $("imageDir").value = typeof sel === "string" ? sel : (sel.path ?? String(sel));
  } catch (e) {}
};
$("imageLoadBtn").onclick = async () => {
  const dir = $("imageDir").value.trim();
  if (!dir) { $("imageModelStatus").textContent = "pick the image_weights directory first"; return; }
  $("imageModelStatus").textContent = "loading image model (mmapping ~2.5 GB)…";
  try {
    const info = await invoke("load_image_model", { dir });
    $("imageModelStatus").textContent = `image model ready · ${info.resolution}px`;
  } catch (e) {
    $("imageModelStatus").textContent = "load failed: " + e;
  }
};
$("imageGenBtn").onclick = async () => {
  if (imageBusy) return;
  const prompt = $("imagePrompt").value.trim();
  if (!prompt) { $("imageProgress").textContent = "type a prompt"; return; }
  const neg = $("imageNeg").value;
  const steps = parseInt($("imageSteps").value) || 50;
  const cfg = parseFloat($("imageCfg").value) || 5;
  const seed = parseInt($("imageSeed").value) || 1234;
  imageBusy = true;
  $("imageGenBtn").disabled = true;
  $("imageResult").innerHTML = "";
  $("imageProgress").textContent = "sampling…";
  try {
    await invoke("generate_image", { prompt, neg, steps, cfg, seed });
  } catch (e) {
    $("imageProgress").textContent = "error: " + e;
    imageBusy = false; $("imageGenBtn").disabled = false;
  }
};
listen("image_progress", (e) => {
  const { step, total } = e.payload || {};
  $("imageProgress").textContent = `denoising step ${step}/${total}… (decoding follows; CPU render is slow)`;
});
listen("image_done", (e) => {
  const { data_uri, w, h } = e.payload || {};
  $("imageProgress").textContent = `done · ${w}×${h}`;
  $("imageResult").innerHTML = `<img src="${data_uri}" alt="generated" style="max-width:100%;border-radius:10px;margin-top:8px" />`;
  imageBusy = false; $("imageGenBtn").disabled = false;
});
