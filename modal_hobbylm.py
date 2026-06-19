"""HobbyLM release pipeline: convert trained MoE checkpoints -> safetensors (per-model HF repos) + GGUF
(arch=hobbylm, one combined repo), with production READMEs. Runs on Modal (weights live on the volumes;
HF auth via the `huggingface` secret). Uploads to PRIVATE rootxhacker/HobbyLM-* repos.

  python -m modal run modal_hobbylm.py --action one --key chat      # smoke-test a single model
  python -m modal run modal_hobbylm.py --action all                 # all LLMs
  python -m modal run modal_hobbylm.py --action image               # HobbyLM-Image (safetensors only)
  python -m modal run modal_hobbylm.py --action gguf-readme         # push the combined GGUF repo README
"""
import modal

app = modal.App("hobbylm-release")

HF_USER = "rootxhacker"
GGUF_REPO = f"{HF_USER}/HobbyLM-gguf"
PRIVATE = True

_ASSETS = "hobby-chat/src-tauri/assets"
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch", "numpy", "safetensors", "gguf>=0.10.0", "huggingface_hub>=0.25.0")
       .add_local_file("to_gguf.py", "/root/moe-lab/to_gguf.py")
       .add_local_file(f"{_ASSETS}/vision_projector.safetensors", "/root/moe-lab/assets/vision_projector.safetensors")
       .add_local_file(f"{_ASSETS}/speech_projector.safetensors", "/root/moe-lab/assets/speech_projector.safetensors")
       .add_local_file(f"{_ASSETS}/melfilters.bytes", "/root/moe-lab/assets/melfilters.bytes"))

runs_vol = modal.Volume.from_name("fineweb10B")
dream_vol = modal.Volume.from_name("dreamlite-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")

# key -> (checkpoint path on /data, repo suffix, human title, one-line summary, pipeline_tag)
LLM_MODELS = {
    "base":         ("/data/runs/500M_ctx8k/model.pt",          "HobbyLM-Base",
                     "HobbyLM-Base (500M sparse-MoE foundation LM)",
                     "A 500M-parameter sparse Mixture-of-Experts base language model pretrained on FineWeb.",
                     "text-generation"),
    "chat":         ("/data/runs/500M_chat_v2/model.pt",        "HobbyLM-Chat",
                     "HobbyLM-Chat (500M MoE, instruction-tuned)",
                     "Conversational/instruction-tuned variant of HobbyLM-Base (SmolTalk-style SFT).",
                     "text-generation"),
    "computer-use": ("/data/runs/500M_computer_use_v4/model.pt", "HobbyLM-Computer-Use",
                     "HobbyLM-Computer-Use (500M MoE, GUI agent / tool use)",
                     "Function-calling + accessibility-tree GUI-agent variant for computer-use tasks.",
                     "text-generation"),
    "omni":         ("/data/runs/500M_vlm_joint12/model.pt",    "HobbyLM-Omni",
                     "HobbyLM-Omni (500M MoE, text + image + audio)",
                     "Multimodal (omni) variant: a TinyLLaVA-style VLM over the HobbyLM MoE core, with vision and speech projectors.",
                     "image-text-to-text"),
    "diffusion":    ("/data/runs/500M_diff_20b/model.pt",       "HobbyLM-Diffusion",
                     "HobbyLM-Diffusion (500M MoE, text diffusion / LLaDA-style)",
                     "Masked-diffusion (LLaDA-style) variant of HobbyLM for bidirectional / parallel decoding.",
                     "text-generation"),
}

ARCH = """## Architecture

HobbyLM is a **sparse Mixture-of-Experts (MoE)** transformer (DeepSeek-V3-style):

| Component | Value |
|---|---|
| Total parameters | ~500M (≈ a fraction active per token) |
| Hidden size / layers | 768 / 16 (1 dense FFN layer, 15 MoE) |
| Routed experts / active | 36 / top-6 (+ 1 always-on shared expert) |
| Attention | GQA, 12 query / 3 KV heads, head-dim 128, per-head QK-norm |
| Router | sigmoid gating, aux-loss-free balancing bias, no top-k renorm |
| Positional | RoPE |
| Tokenizer | GPT-2 byte-level BPE (50,304 vocab, sentinel-padded) |
"""


def _readme(title, summary, pipeline_tag, key):
    extra = ""
    if key == "omni":
        extra = ("\n## Multimodal use\n\nThis repo also ships the projector weights:\n"
                 "`vision_projector.safetensors` (SigLIP2 → LLM) and `speech_projector.safetensors` "
                 "(Whisper-mel → LLM), plus `melfilters.bytes`. Image/audio are encoded by the (frozen) "
                 "SigLIP2 / mel front-ends, projected, and spliced in at the `[IMAGE]`/`[AUDIO]`/`[SPEECH]` "
                 "sentinel tokens (ids 50257–50262).\n")
    if key == "diffusion":
        extra = ("\n## Decoding\n\nThis is a **masked-diffusion** checkpoint (LLaDA-style): generation is "
                 "iterative bidirectional denoising of `[MASK]` tokens, not left-to-right AR. The GGUF carries "
                 "`diffusion.*` metadata (mask token id, block size) for a diffusion-aware runtime.\n")
    return f"""---
license: apache-2.0
language: [en]
library_name: safetensors
pipeline_tag: {pipeline_tag}
tags: [hobbylm, mixture-of-experts, moe, sparse-moe]
---

# {title}

{summary}

Part of the **HobbyLM** family — a from-scratch 500M sparse-MoE model trained on consumer-scale budgets.

{ARCH}
{extra}
## Files

- `model.safetensors` — the model weights (fp32).
- `config.json` — architecture / hyperparameters.
- GGUF builds (arch `hobbylm`) live in [`{GGUF_REPO}`](https://huggingface.co/{GGUF_REPO}).

## Loading (safetensors)

```python
import json, torch
from safetensors.torch import load_file
sd  = load_file("model.safetensors")
cfg = json.load(open("config.json"))
# rebuild the HobbyLM nn.Module from `cfg` and `load_state_dict(sd)`.
```

## Notes & limitations

- Research model at the ~500M scale: fluent but with the capability ceiling of a small model.
- The GGUF uses a custom `hobbylm` architecture (see the GGUF repo) and needs `hobby-rs` or a patched llama.cpp.

## License

Apache-2.0.
"""


def _save_safetensors(sd, path):
    import torch
    from safetensors.torch import save_file
    clean = {k: v.detach().to(torch.float32).cpu().contiguous().clone()
             for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(clean, path)


@app.function(image=img, volumes={"/data": runs_vol}, secrets=[HF_SECRET], timeout=60 * 60, memory=24000)
def export_llm(key: str):
    import os, sys, json, subprocess, torch
    from huggingface_hub import HfApi
    sys.path.insert(0, "/root/moe-lab"); os.chdir("/root/moe-lab")
    ckpt, suffix, title, summary, ptag = LLM_MODELS[key]
    repo = f"{HF_USER}/{suffix}"
    api = HfApi()
    print(f"[{key}] loading {ckpt}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd, cfg = ck["model"], ck["config"]

    os.makedirs("/tmp/out", exist_ok=True)
    _save_safetensors(sd, "/tmp/out/model.safetensors")
    json.dump(cfg, open("/tmp/out/config.json", "w"), indent=2, default=str)
    open("/tmp/out/README.md", "w").write(_readme(title, summary, ptag, key))

    # ---- GGUF (arch=hobbylm) ----
    gguf_name = f"{suffix}.gguf"
    subprocess.run([sys.executable, "to_gguf.py", "--ckpt", ckpt, "--out", f"/tmp/out/{gguf_name}",
                    "--arch", "hobbylm"], check=True)

    # ---- per-model safetensors repo ----
    api.create_repo(repo, private=PRIVATE, exist_ok=True, repo_type="model")
    for f in ["model.safetensors", "config.json", "README.md"]:
        api.upload_file(path_or_fileobj=f"/tmp/out/{f}", path_in_repo=f, repo_id=repo, repo_type="model")
    # ---- omni: ship projectors too ----
    if key == "omni":
        for pf in ["assets/vision_projector.safetensors", "assets/speech_projector.safetensors", "assets/melfilters.bytes"]:
            if os.path.exists(pf):
                api.upload_file(path_or_fileobj=pf, path_in_repo=os.path.basename(pf), repo_id=repo, repo_type="model")

    # ---- combined GGUF repo ----
    api.create_repo(GGUF_REPO, private=PRIVATE, exist_ok=True, repo_type="model")
    api.upload_file(path_or_fileobj=f"/tmp/out/{gguf_name}", path_in_repo=gguf_name, repo_id=GGUF_REPO, repo_type="model")
    sz = os.path.getsize(f"/tmp/out/{gguf_name}") / 1e9
    print(f"[{key}] DONE -> {repo} (safetensors) + {GGUF_REPO}/{gguf_name} ({sz:.2f} GB)", flush=True)
    return {"key": key, "repo": repo, "gguf": gguf_name}


@app.function(image=img, volumes={"/cache": dream_vol}, secrets=[HF_SECRET], timeout=30 * 60, memory=16000)
def export_image():
    import os, json, torch
    from huggingface_hub import HfApi
    repo = f"{HF_USER}/HobbyLM-Image"
    api = HfApi()
    ck = torch.load("/cache/model_1024flux4.pt", map_location="cpu", weights_only=False)
    sd, cfgd = ck["sd"], ck["cfg_dict"]
    os.makedirs("/tmp/img", exist_ok=True)
    _save_safetensors(sd, "/tmp/img/model.safetensors")
    json.dump({"dit_config": cfgd, "lat_std": float(ck["lat_std"]), "scaling_factor": float(ck["sf"]),
               "vae": "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
               "text_encoder": "openai/clip-vit-large-patch14", "resolution": 1024,
               "steps_trained": int(ck.get("steps", 0))}, open("/tmp/img/config.json", "w"), indent=2)
    open("/tmp/img/README.md", "w").write(f"""---
license: apache-2.0
pipeline_tag: text-to-image
library_name: safetensors
tags: [hobbylm, text-to-image, diffusion, dit, flow-matching]
---

# HobbyLM-Image (1024px text-to-image DiT)

An in-context latent **flow-matching DiT** that generates 1024×1024 images, trained on a $300-class budget.
It operates in the **DC-AE f32c32 (SANA-1.1)** latent space and is conditioned on **CLIP-L** text features.

## Components (frozen, not included)

- VAE: `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` (32× spatial compression → 32×32×32 latent at 1024px).
- Text encoder: `openai/clip-vit-large-patch14`.

## Files
- `model.safetensors` — the DiT weights. `config.json` — DiT config, `lat_std`, VAE `scaling_factor`.

## Pipeline (sketch)
Encode the text prompt with CLIP-L → start from Gaussian latent noise → run the DiT's rectified-flow / CFG
sampler for ~100 steps → decode the latent with the DC-AE VAE → 1024px image. (No GGUF: image-gen DiTs have
no standard GGUF runtime.)

## Capabilities
Watermark-free; accurate objects; cinematic scenes; usable single-person portraits. Soft on hands /
multi-person (the small-model ceiling). Editing is available in a sibling 512px checkpoint.

## License
Apache-2.0.
""")
    api.create_repo(repo, private=PRIVATE, exist_ok=True, repo_type="model")
    for f in ["model.safetensors", "config.json", "README.md"]:
        api.upload_file(path_or_fileobj=f"/tmp/img/{f}", path_in_repo=f, repo_id=repo, repo_type="model")
    print(f"image DONE -> {repo}", flush=True)
    return {"repo": repo}


@app.function(image=img, secrets=[HF_SECRET], timeout=10 * 60)
def push_gguf_readme():
    from huggingface_hub import HfApi
    api = HfApi()
    body = f"""---
license: apache-2.0
tags: [hobbylm, gguf, mixture-of-experts, moe]
---

# HobbyLM-GGUF

GGUF builds of every **HobbyLM** language model, one file per variant:

| File | Model |
|---|---|
| `HobbyLM-Base.gguf` | foundation LM |
| `HobbyLM-Chat.gguf` | instruction / chat |
| `HobbyLM-Computer-Use.gguf` | GUI agent / tool use |
| `HobbyLM-Omni.gguf` | multimodal core (text+image+audio) |
| `HobbyLM-Diffusion.gguf` | text-diffusion (LLaDA-style) |

## ⚠️ Architecture: `hobbylm`

These GGUFs set `general.architecture = hobbylm` (all KV keys are `hobbylm.*`). **Stock llama.cpp will not
load them** — they require the **`hobby-rs`** engine or a llama.cpp patched to register the `hobbylm` arch
(GQA + per-head QK-norm + sigmoid-gated MoE + aux-free bias + 1 shared expert + leading dense layer). The
`HobbyLM-Diffusion` build additionally carries `diffusion.*` metadata and needs a diffusion-aware decoder.

## License
Apache-2.0.
"""
    api.create_repo(GGUF_REPO, private=PRIVATE, exist_ok=True, repo_type="model")
    open("/tmp/gr.md", "w").write(body)
    api.upload_file(path_or_fileobj="/tmp/gr.md", path_in_repo="README.md", repo_id=GGUF_REPO, repo_type="model")
    print(f"pushed README -> {GGUF_REPO}", flush=True)


@app.local_entrypoint()
def main(action: str = "one", key: str = "chat"):
    if action == "one":
        print(export_llm.remote(key))
    elif action == "all":
        handles = {k: export_llm.spawn(k) for k in LLM_MODELS}      # all LLMs in parallel
        handles["image"] = export_image.spawn()                     # + the image model
        for k, h in handles.items():
            try:
                print(h.get())
            except Exception as e:
                print(f"{k} FAILED: {e}")
        print(push_gguf_readme.remote())
    elif action == "image":
        print(export_image.remote())
    elif action == "gguf-readme":
        print(push_gguf_readme.remote())
