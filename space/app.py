"""HobbyLM Playground — a Gradio Space to chat with the HobbyLM models, ask questions about an
image (the multimodal Omni model), and generate images (the 1024px DiT + DC-AE pipeline).

All models are the from-scratch 500M sparse-MoE family (+ a 333M image DiT) published at
https://huggingface.co/rootxhacker . They use a custom architecture, so the Space vendors the
reference implementation (`hobbylm/`, `hobby_image/`) instead of going through transformers' AutoModel.

Runs on ZeroGPU (the heavy functions are @spaces.GPU); falls back to CPU when run locally.
"""
import json
import threading

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# --- Work around a long-standing gradio_client bug ("argument of type 'bool' is not iterable" in
# get_type / json_schema_to_python_type when a component schema has a boolean `additionalProperties`).
# It crashes the /info endpoint, so the Gradio frontend shows "No API found" and can't call functions.
# Treat boolean schemas as `Any`. (Present in both gradio 4.44 and 5.9's bundled gradio_client.)
import gradio_client.utils as _gcu  # noqa: E402

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

_orig_jstpt = _gcu._json_schema_to_python_type
def _safe_jstpt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_jstpt(schema, defs)
_gcu._json_schema_to_python_type = _safe_jstpt

# ZeroGPU decorator — with a no-op fallback so the app also runs on plain CPU / locally.
try:
    import spaces
except Exception:  # not on a ZeroGPU Space
    class _Spaces:
        @staticmethod
        def GPU(*a, **k):
            if a and callable(a[0]):
                return a[0]
            def deco(f):
                return f
            return deco
    spaces = _Spaces()

HF_USER = "rootxhacker"
VISION_ID = "google/siglip2-so400m-patch16-512"   # the encoder HobbyLM-Omni was trained with
DCAE_ID = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
CLIP_ID = "openai/clip-vit-large-patch14"
NEG_DEFAULT = "blurry, low quality, watermark, signature, text, jpeg artifacts, deformed, distorted"

# chat dropdown -> (repo suffix, decoding kind)
CHAT_MODELS = {
    "HobbyLM-Chat — instruction / conversation": ("HobbyLM-Chat", "chat"),
    "HobbyLM-Base — raw text completion": ("HobbyLM-Base", "base"),
    "HobbyLM-Computer-Use — tools / GUI agent": ("HobbyLM-Computer-Use", "chat"),
    "HobbyLM-Omni — multimodal core (text)": ("HobbyLM-Omni", "chat"),
    "HobbyLM-Diffusion — masked-diffusion LM": ("HobbyLM-Diffusion", "diffusion"),
}

DEFAULT_CHAT = list(CHAT_MODELS)[0]

_cache = {}
_lock = threading.Lock()


def _warmup():
    """Build the heavy models in the MAIN process at startup. ZeroGPU runs each @spaces.GPU call in a
    forked worker that inherits the main process's memory, so models built here are reused across calls
    (no per-call rebuild) — which is what was blowing the Omni GPU-time budget. Chat LLMs stay lazy
    (they're light enough to rebuild per call). Runs in a daemon thread so the app binds the port now."""
    try:
        from huggingface_hub import snapshot_download
        for mid in [VISION_ID, DCAE_ID, CLIP_ID]:
            snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        _load_vlm()             # Omni LLM + SigLIP2 + projector (the expensive one for the image tab)
        _load_image_models()    # DiT + DC-AE + CLIP
        print("[warmup] VLM + image models built in main process", flush=True)
    except Exception as e:
        print(f"[warmup] warning: {e}", flush=True)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


# --------------------------------------------------------------------------- loaders (cached)

# NOTE: ZeroGPU releases/re-attaches the GPU between calls, so models are cached on **CPU** and moved
# to CUDA *inside* each @spaces.GPU call (then back to CPU) — caching a model on CUDA and reusing it
# across calls crashes the ZeroGPU worker.

# Loaders are LOCK-FREE: Gradio serializes requests (concurrency 1), and a lock held during a slow
# build would deadlock a ZeroGPU fork. Dict get/set is atomic under the GIL.

def _load_llm(repo):
    key = ("llm", repo)
    if key in _cache:
        return _cache[key]
    from hobbylm.config import ModelConfig
    from hobbylm.model import MoETransformer
    cfg_d = {k: v for k, v in json.load(
        open(hf_hub_download(f"{HF_USER}/{repo}", "config.json"))).items() if k != "preset"}
    cfg = ModelConfig(**cfg_d)
    cfg.expert_backend = "bmm"                      # universal MoE backend (CPU + GPU)
    model = MoETransformer(cfg).eval()
    model.load_state_dict(load_file(hf_hub_download(f"{HF_USER}/{repo}", "model.safetensors")))
    _cache[key] = (model, cfg)
    return _cache[key]


def _load_vlm():
    key = ("vlm",)
    if key in _cache:
        return _cache[key]
    from hobbylm.vision import SiglipVision
    from hobbylm.multimodal import MoEVLM
    llm, _ = _load_llm("HobbyLM-Omni")
    enc = SiglipVision(model_id=VISION_ID, device="cpu", dtype=torch.float32)
    vlm = MoEVLM(llm, vision_dim=enc.hidden)
    vlm.mm_projector.load_state_dict(
        load_file(hf_hub_download(f"{HF_USER}/HobbyLM-Omni", "vision_projector.safetensors")))
    vlm.eval()
    _cache[key] = (vlm, enc)
    return _cache[key]


def _load_image_models():
    if ("dit",) not in _cache:
        from hobby_image.dit import HobbyImageDiT, DiTConfig
        cfg = json.load(open(hf_hub_download(f"{HF_USER}/HobbyLM-Image", "config.json")))
        dit = HobbyImageDiT(DiTConfig(**cfg["dit_config"])).eval()
        dit.load_state_dict(load_file(hf_hub_download(f"{HF_USER}/HobbyLM-Image", "model.safetensors")))
        _cache[("dit",)] = (dit, cfg["dit_config"]["latent_h"], float(cfg["lat_std"]), float(cfg["scaling_factor"]))
    if ("dcae",) not in _cache:
        from diffusers import AutoencoderDC
        # bf16 (NOT fp16): the DiT/DC-AE overflow in fp16 -> NaN -> black images.
        _cache[("dcae",)] = AutoencoderDC.from_pretrained(DCAE_ID, torch_dtype=torch.bfloat16).eval()
    if ("clip",) not in _cache:
        from transformers import CLIPTextModel, CLIPTokenizer
        _cache[("clip",)] = (CLIPTokenizer.from_pretrained(CLIP_ID),
                             CLIPTextModel.from_pretrained(CLIP_ID, torch_dtype=torch.bfloat16).eval())
    dit, lat, lat_std, sf = _cache[("dit",)]
    ae = _cache[("dcae",)]
    tok, clip = _cache[("clip",)]
    return dit, lat, lat_std, sf, ae, tok, clip


SAE_REPO = "rootxhacker/HobbyLM-SAE"


def _load_sae():
    key = ("sae",)
    if key in _cache:
        return _cache[key]
    import json
    from hobbylm.sae import TopKSAE, SAEConfig
    meta = json.load(open(hf_hub_download(SAE_REPO, "meta.json")))
    labels = json.load(open(hf_hub_download(SAE_REPO, "labels.json")))
    sae = TopKSAE(SAEConfig(**meta["cfg"])).eval()
    sae.load_state_dict(load_file(hf_hub_download(SAE_REPO, "sae.safetensors")))
    _cache[key] = (sae, meta, labels)
    return _cache[key]


# --------------------------------------------------------------------------- chat

def _build_prompt(repo, message, history):
    if repo == "HobbyLM-Base":
        return message                                     # base = pure completion
    s = ""
    for turn in history or []:
        if isinstance(turn, dict):                         # gradio 5 "messages" format
            role, content = turn.get("role"), turn.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if role == "user":
                s += f"USER: {content}\n"
            elif role == "assistant" and content:
                s += f"ASSISTANT: {content}\n"
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:  # legacy "tuples" format
            u, a = turn
            if u:
                s += f"USER: {u}\n"
            if a:
                s += f"ASSISTANT: {a}\n"
    return s + f"USER: {message}\nASSISTANT:"


@spaces.GPU(duration=180)
def chat_fn(message, history, model_name, max_new, temperature):
    from hobbylm.generate import generate as ar_generate
    repo, kind = CHAT_MODELS[model_name]
    dev = _device()
    enc = _enc()
    prompt = _build_prompt(repo, message, history)
    model, cfg = _load_llm(repo)
    model.to(dev)
    try:
        ids = torch.tensor([enc.encode_ordinary(prompt)], device=dev)
        if kind == "diffusion":
            from hobbylm.diffusion import generate as dgen
            gen_len = int(max_new)
            out = dgen(model, ids, gen_len=gen_len, steps=max(32, 2 * gen_len),
                       temperature=max(0.0, float(temperature) - 0.4), rep_penalty=1.5, remask_steps=2)
            return enc.decode(out[0].tolist()).strip()
        ctx_len = min(getattr(cfg, "context_length", 1024), 2048)
        out = ar_generate(model, ids, int(max_new), float(temperature), 0, torch.device(dev),
                          top_p=0.95, repetition_penalty=1.3, no_repeat_ngram_size=3, ctx_len=ctx_len)
        return enc.decode(out[0, ids.shape[1]:].tolist()).strip()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ error: {e}"
    finally:
        model.to("cpu")


# --------------------------------------------------------------------------- image understanding (Omni)

@spaces.GPU(duration=180)
def understand_fn(image, question, max_new):
    if image is None:
        return "Please upload an image first."
    from hobbylm.multimodal import IMAGE_TOKEN
    from hobbylm.generate import GPT2_VALID, EOT
    dev = _device()
    enc = _enc()
    vlm, venc = _load_vlm()
    vlm.to(dev)
    venc.vision.to(dev)
    venc.device = dev
    try:
        from contextlib import nullcontext
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()
        with torch.no_grad(), amp:
            feats = venc.encode([image.convert("RGB")]).float()
            q = (question or "Describe this image in detail.").strip()
            pre = enc.encode_ordinary(f"USER: {q}\nASSISTANT:")
            ids = torch.tensor([[IMAGE_TOKEN] + pre], device=dev)
            cur, _ = vlm.build_inputs_embeds(ids, image_features=feats)
            outs = []
            for _ in range(int(max_new)):
                logits, _ = vlm.llm(inputs_embeds=cur)
                lg = logits[:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:                                    # repetition penalty
                    u = torch.tensor(sorted(set(outs)), device=dev)
                    v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                e = vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)
                cur = torch.cat([cur, e], dim=1)
        return enc.decode(outs).strip() or "(no answer)"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"⚠️ error: {e}"
    finally:
        vlm.to("cpu")
        venc.vision.to("cpu")
        venc.device = "cpu"


# --------------------------------------------------------------------------- image generation

@spaces.GPU(duration=180)
def generate_image_fn(prompt, negative, steps, guidance, seed, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    from PIL import Image
    import numpy as np
    from contextlib import nullcontext
    dev = _device()
    dit, lat, lat_std, sf, ae, tok, clip = _load_image_models()
    dit.to(dev)
    ae.to(dev)
    clip.to(dev)
    steps = int(steps)
    neg = (negative or "").strip()

    def clip_encode(texts):
        ids = tok(texts, padding="max_length", max_length=64, truncation=True,
                  return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            return clip(ids).last_hidden_state.float()

    try:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        ctx = clip_encode([prompt])
        uncond = clip_encode([neg]) if neg else torch.zeros_like(ctx)
        task = torch.zeros(1, dtype=torch.long, device=dev)
        z = torch.randn(1, 32, lat, lat, generator=g, device=dev)
        zs = torch.zeros(1, 32, lat, lat, device=dev)
        em = torch.zeros(1, 1, lat, 2 * lat, device=dev)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()
        ae_dtype = next(ae.parameters()).dtype
        with torch.no_grad():
            for i in progress.tqdm(range(steps), desc="denoising"):
                tt = torch.full((1,), i / steps, device=dev)
                inp = torch.cat([torch.cat([z, zs], dim=-1), em, em], dim=1)
                with amp:
                    vc = dit(inp, tt, ctx, task)[..., :lat].float()
                    vu = dit(inp, tt, uncond, task)[..., :lat].float()
                z = z + (vu + float(guidance) * (vc - vu)) / steps
            with amp:
                img = ae.decode((z * lat_std / sf).to(ae_dtype)).sample
        img = img.float().clamp(-1, 1)[0]
        arr = ((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    finally:
        dit.to("cpu")
        ae.to("cpu")
        clip.to("cpu")


# Pre-build the heavy models in the MAIN process, in a background thread (non-blocking startup). The
# Omni VLM was crashing because building it *inside* the GPU window blew the time limit and the worker
# was killed before the result could cache — so it rebuilt and died every call. Building here once means
# ZeroGPU workers inherit it and only do (fast) inference. Lock-free loaders => no fork-while-locked hang.
threading.Thread(target=_warmup, daemon=True).start()


# --------------------------------------------------------------------------- how it works (MoE routing)

@spaces.GPU(duration=90)
def how_it_works(prompt, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    dev = _device()
    enc = _enc()
    model, cfg = _load_llm("HobbyLM-Base")
    model.to(dev)
    try:
        ids = enc.encode_ordinary(prompt or "The quick brown fox jumps over the lazy dog.")[:40]
        if not ids:
            ids = enc.encode_ordinary("Hello world")
        toks = torch.tensor([ids], device=dev)
        with torch.no_grad():
            model(toks)                                    # populates last_topi on each MoE block
        ne, S = cfg.n_experts, len(ids)
        moe_layers = [i for i, b in enumerate(model.blocks) if getattr(b, "is_moe", False)]
        layer = min(max(int(layer), moe_layers[0]), moe_layers[-1])
        blk = model.blocks[layer]
        topi = blk.ffn.last_topi.reshape(S, -1).cpu().numpy()
        topv = blk.ffn.last_topv.reshape(S, -1).cpu().float().numpy()
        labels = [repr(enc.decode([i]))[1:-1][:12] for i in ids]

        # (1) per-token routing heatmap at the chosen layer
        M = np.zeros((S, ne))
        for s in range(S):
            for j in range(topi.shape[1]):
                M[s, int(topi[s, j])] = topv[s, j]
        fig1, ax = plt.subplots(figsize=(11, max(2.5, S * 0.32)))
        im = ax.imshow(M, aspect="auto", cmap="magma")
        ax.set_yticks(range(S)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"expert (0–{ne - 1})"); ax.set_ylabel("token")
        ax.set_title(f"Layer {layer}: each token routes to its top-{cfg.top_k} of {ne} experts (+1 shared, always on)")
        fig1.colorbar(im, ax=ax, label="gate weight", fraction=0.025)
        fig1.tight_layout()

        # (2) expert load across ALL MoE layers (the load-balancing story)
        load = np.zeros(ne)
        for i in moe_layers:
            for e in model.blocks[i].ffn.last_topi.reshape(-1).cpu().numpy():
                load[int(e)] += 1
        fig2, ax2 = plt.subplots(figsize=(11, 2.6))
        ax2.bar(range(ne), load, color="#7c3aed")
        ax2.set_xlabel("expert"); ax2.set_ylabel("tokens routed")
        ax2.set_title(f"Expert load over all {len(moe_layers)} MoE layers — fairly even = aux-loss-free balancing working")
        fig2.tight_layout()

        active = cfg.top_k + cfg.n_shared
        summary = (f"**{S} tokens** · **{ne} experts/layer**, top-{cfg.top_k} routed + {cfg.n_shared} shared. "
                   f"At each of the {len(moe_layers)} MoE layers every token uses only **{active}/{ne + cfg.n_shared} "
                   f"experts** → that's the *sparse* in sparse-MoE: a 500M model that computes like a far smaller one "
                   f"per token. Different tokens pick different experts (the heatmap); across the whole prompt the load "
                   f"spreads fairly evenly (the bar chart).")
        return fig1, fig2, summary
    finally:
        model.to("cpu")


# --------------------------------------------------------------------------- how it works (SAE features)

@spaces.GPU(duration=90)
def sae_features(prompt, topn):
    dev = _device()
    enc = _enc()
    try:
        sae, meta, labels = _load_sae()
    except Exception as e:
        return f"⚠️ SAE not available yet: {e}"
    model, _ = _load_llm("HobbyLM-Base")
    model.to(dev); sae.to(dev)
    layer, scale = meta["layer"], float(meta["scale"])
    topn = int(topn)
    try:
        ids = enc.encode_ordinary(prompt or "I love listening to music while coding software.")[:48]
        if not ids:
            ids = enc.encode_ordinary("Hello world")
        toks = torch.tensor([ids], device=dev)
        with torch.no_grad():
            h = model(toks, capture_layer=layer).float() * scale
            z = sae.encode(h.reshape(-1, sae.cfg.d_in))           # (S, m)
        md = ("Each token's residual is decomposed into a few **interpretable features** from the SAE "
              "dictionary. Below: per token, the strongest features (auto-labelled by the tokens they "
              "fire on most).\n\n| token | top active features &nbsp;·&nbsp; *(label · strength)* |\n|---|---|\n")
        for s, tid in enumerate(ids):
            v, f = z[s].topk(min(topn, z.shape[1]))
            tok_str = enc.decode([tid]).replace("|", "¦").replace("\n", "⏎").strip() or "·"
            parts = []
            for val, fi in zip(v.tolist(), f.tolist()):
                if val <= 1e-4:
                    continue
                lab = labels.get(str(int(fi)), {}).get("label") or f"feat#{int(fi)}"
                parts.append(f"**{lab}** ({val:.1f})")
            md += f"| `{tok_str}` | {' · '.join(parts) or '—'} |\n"
        return md
    finally:
        model.to("cpu"); sae.to("cpu")


# --------------------------------------------------------------------------- UI

INTRO = """# 🪶 HobbyLM Playground

A from-scratch **500M sparse Mixture-of-Experts** model family (+ a 333M image DiT), trained on a hobby
budget. Chat with any variant, ask questions about an image with the multimodal **Omni** model, or
generate a 1024px image. Models: [rootxhacker on Hugging Face](https://huggingface.co/rootxhacker) ·
code: [GitHub](https://github.com/harishsg993010/HobbyLM).

*These are tiny research models — fluent and fun, with the capability ceiling of a 500M model.*
"""

with gr.Blocks(title="HobbyLM Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("💬 Chat"):
        model_dd = gr.Dropdown(list(CHAT_MODELS), value=DEFAULT_CHAT, label="Model")
        with gr.Row():
            max_new = gr.Slider(16, 512, value=200, step=8, label="Max new tokens")
            temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature (0 = greedy)")
        gr.ChatInterface(
            fn=chat_fn,
            type="messages",
            additional_inputs=[model_dd, max_new, temp],
            # with additional_inputs, each example row is [message, model, max_new, temp]
            examples=[["Give me three tips for better sleep.", DEFAULT_CHAT, 200, 0.7],
                      ["Explain a mixture-of-experts model in one sentence.", DEFAULT_CHAT, 200, 0.7],
                      ["Write a short poem about the ocean.", DEFAULT_CHAT, 200, 0.7]],
            cache_examples=False,
        )

    with gr.Tab("🖼️ Ask about an image"):
        gr.Markdown("Upload an image and ask the **HobbyLM-Omni** multimodal model about it.")
        with gr.Row():
            with gr.Column():
                u_img = gr.Image(type="pil", label="Image")
                u_q = gr.Textbox(label="Question", value="Describe this image in detail.")
                u_max = gr.Slider(16, 128, value=48, step=8, label="Max new tokens")
                u_btn = gr.Button("Ask", variant="primary")
            u_out = gr.Textbox(label="Answer", lines=6)
        u_btn.click(understand_fn, [u_img, u_q, u_max], u_out)

    with gr.Tab("🎨 Generate an image"):
        gr.Markdown("Text-to-image with **HobbyLM-Image** (1024px DiT in DC-AE latent space). "
                    "Strongest on single objects and cinematic scenes.")
        with gr.Row():
            with gr.Column():
                g_prompt = gr.Textbox(label="Prompt", value="a red convertible car on a coastal road, golden hour")
                g_neg = gr.Textbox(label="Negative prompt", value=NEG_DEFAULT)
                with gr.Row():
                    g_steps = gr.Slider(20, 120, value=60, step=5, label="Steps")
                    g_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="Guidance (CFG)")
                    g_seed = gr.Number(value=1234, label="Seed", precision=0)
                g_btn = gr.Button("Generate", variant="primary")
            g_out = gr.Image(label="Result", height=512)
        g_btn.click(generate_image_fn, [g_prompt, g_neg, g_steps, g_cfg, g_seed], g_out)
        gr.Examples([["a photograph of a single red apple on a plain white background", NEG_DEFAULT, 60, 5.0, 1234],
                     ["a cozy library with tall wooden bookshelves, warm light", NEG_DEFAULT, 80, 5.0, 7],
                     ["a bowl of fresh strawberries, studio food photography", NEG_DEFAULT, 60, 5.0, 42]],
                    [g_prompt, g_neg, g_steps, g_cfg, g_seed], cache_examples=False)

    with gr.Tab("🔬 How it works"):
        gr.Markdown(
            "HobbyLM is a **sparse Mixture-of-Experts**: each MoE layer holds **36 little expert networks**, "
            "but a router sends every token through only its **top-6** (plus 1 always-on shared expert). "
            "So a 500M model does the *compute* of a much smaller one per token. Type some text and watch the "
            "router decide — which experts each token uses, and how the load spreads across all 36.")
        with gr.Row():
            hiw_prompt = gr.Textbox(label="Text", value="The capital of France is Paris, a beautiful city.", scale=4)
            hiw_layer = gr.Slider(1, 15, value=8, step=1, label="MoE layer", scale=1)
        hiw_btn = gr.Button("Visualize routing", variant="primary")
        hiw_summary = gr.Markdown()
        hiw_heat = gr.Plot(label="Per-token expert routing")
        hiw_load = gr.Plot(label="Expert load (balancing)")
        hiw_btn.click(how_it_works, [hiw_prompt, hiw_layer], [hiw_heat, hiw_load, hiw_summary])

    with gr.Tab("🧠 What it represents"):
        gr.Markdown(
            "A **sparse autoencoder** (SAE) trained on HobbyLM-Base's layer-8 residual stream pulls apart each "
            "activation into a handful of **interpretable features** from a 12,288-entry dictionary. Type text and "
            "see which concepts light up on each token — words, synonym clusters, syntax, formatting. This is "
            "*mechanistic interpretability*: looking at what the model actually represents inside.")
        sae_prompt = gr.Textbox(label="Text", value="I love listening to music while coding software.")
        sae_top = gr.Slider(2, 8, value=4, step=1, label="Features shown per token")
        sae_btn = gr.Button("Show features", variant="primary")
        sae_out = gr.Markdown()
        sae_btn.click(sae_features, [sae_prompt, sae_top], sae_out)

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
