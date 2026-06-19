"""Modal harness for the multimodal MoE-VLM (image + audio). TinyLLaVA-style.

  # download the LLaVA-Pretrain (LAION-CC-SBU-558K) alignment data to a volume:
  python -m modal run modal_mm.py --action download

  # GPU smoke: real SigLIP2 + 500M_ctx2048 backbone + MoEVLM forward/backward on a synthetic image:
  python -m modal run modal_mm.py --action smoke
"""
import modal

# Files the Modal functions never need (Rust app + build artifacts + big local data). Excluding them
# keeps the mount tiny + stable (otherwise the multi-GB hobby-chat/target collides with active cargo builds
# -> "modified during build", and re-uploads GBs every run).
_IGNORE = [
    "hobby-chat/**", "hobby-rs/**", "hobby-rs-cli/**", "**/target/**", "**/.git/**", "**/node_modules/**",
    "*.gguf", "*.onnx", "*.onnx.data", "*.dll", "*.bin", "*.wav", "*.safetensors",
]

# vision/LLM deps (shared base). add_local_dir MUST be last, so keep the dep layers separate and
# append the local files at the very end of each concrete image.
_vlm_deps = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "transformers>=4.50,<5", "pillow", "numpy",
                 "huggingface-hub", "accelerate", "sentencepiece", "tiktoken",
                 "soundfile", "librosa", "ijson", "safetensors", "onnx", "onnxscript", "gguf",
                 "datasets==2.21.0", "pyarrow==17.0.0", "pandas==2.2.2")   # pinned set: audio decode via soundfile
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache/hf"})
)
vlm_image = _vlm_deps.add_local_dir(".", "/root/moe-lab", ignore=_IGNORE)

# llama.cpp prebuilt into the image (CPU, portable) for the custom engine: convert -> GGUF -> run in stock llama.cpp
litert_image = (
    _vlm_deps.apt_install("git", "cmake", "build-essential", "libcurl4-openssl-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        # 1-line patch: bailingmoe2 hardcodes Q=n_embd in the fused qkv; support DECOUPLED head_dim
        # (our head_dim=128 != n_embd/n_head). build_qkv already uses n_embd_head*n_head, so only the
        # tensor-creation dim is wrong. Q dim n_embd -> n_embd_head_k*n_head.
        r"""sed -i 's/{ *n_embd, *n_embd *+ *2\*n_embd_gqa *}/{n_embd, n_embd_head_k*n_head + 2*n_embd_gqa}/' /llama.cpp/src/models/bailingmoe2.cpp""",
        "grep -n 'n_embd_head_k\\*n_head + 2\\*n_embd_gqa' /llama.cpp/src/models/bailingmoe2.cpp",  # verify patch applied
        # AUDIO FIX: whisper preprocessor appends a 30s silence pad then splits the mel into complete 3000-frame
        # (30s) chunks -> our <=30s audio spans 2 chunks = 1500 tokens, but our model trained on ONE 30s window
        # (750 tok); the trailing all-silence chunk is OOD -> garbled ASR. Cap the chunk loop to the FIRST chunk
        # (= exactly HF/training: pad/truncate to 30s, 750 tok). GGML_ASSERT(n_len>3000) still holds (+30s pad).
        "echo CnA9Ii9sbGFtYS5jcHAvdG9vbHMvbXRtZC9tdG1kLWF1ZGlvLmNwcCIKcz1vcGVuKHApLnJlYWQoKQphPSIgICAgZm9yIChzaXplX3Qgb2ZmID0gMDsgb2ZmIDwgKHNpemVfdCkgb3V0X2Z1bGwubl9sZW47IG9mZiArPSBmcmFtZXNfcGVyX2NodW5rKSB7IgpiPSIgICAgZm9yIChzaXplX3Qgb2ZmID0gMDsgb2ZmIDwgZnJhbWVzX3Blcl9jaHVuazsgb2ZmICs9IGZyYW1lc19wZXJfY2h1bmspIHsiCmFzc2VydCBzLmNvdW50KGEpPT0xLCAoIkZPUkxPT1AiLHMuY291bnQoYSkpCnM9cy5yZXBsYWNlKGEsYikKb3BlbihwLCJ3Iikud3JpdGUocykKcHJpbnQoIkNIVU5LUEFUQ0ggT0siKQo= | base64 -d | python3 -",
        # DEBUG: dump clip's projected embeddings -> /tmp/clip_embd.bin (compare vs PyTorch speech_projector)
        "echo CnA9Ii9sbGFtYS5jcHAvdG9vbHMvbXRtZC9jbGlwLmNwcCIKcz1vcGVuKHApLnJlYWQoKQphPSJnZ21sX2JhY2tlbmRfdGVuc29yX2dldChlbWJlZGRpbmdzLCB2ZWMsIDAsIGdnbWxfbmJ5dGVzKGVtYmVkZGluZ3MpKTsiCmFzc2VydCBzLmNvdW50KGEpPj0xLCAoIkVNQkQiLHMuY291bnQoYSkpCmQ9YSsnXG4gICAgICAgIHsgc3RkOjpvZnN0cmVhbSBkZigiL3RtcC9jbGlwX2VtYmQuYmluIixzdGQ6Omlvczo6YmluYXJ5KTsgbG9uZyBuYj1nZ21sX25ieXRlcyhlbWJlZGRpbmdzKTsgZGYud3JpdGUoKGNoYXIqKXZlYyxuYik7IH0nCm9wZW4ocCwidyIpLndyaXRlKHMucmVwbGFjZShhLGQsMSkpCnByaW50KCJFTUJEUEFUQ0ggT0siKQo= | base64 -d | python3 -",
        # AUDIO FIX #2: the VOXTRAL audio case never sets n_merge, so build_vit defaults to n_merge=2 and MERGES
        # tokens 1500->750 BEFORE build_stack (->375) = a DOUBLE downsample. Our whisper-small only wants the
        # stack (1500->750). Force n_merge=1 (no token merge) in the audio case.
        "echo CnA9Ii9sbGFtYS5jcHAvdG9vbHMvbXRtZC9jbGlwLmNwcCIKcz1vcGVuKHApLnJlYWQoKQphPScgICAgICAgICAgICAgICAgICAgICAgICBsb2dfZmZuX29wID0gImdlbHVfZXJmIjsgLy8gdGVtcG9yYXJ5IHNvbHV0aW9uIGZvciBsb2dnaW5nJwphc3NlcnQgcy5jb3VudChhKT09MSwgKCJBTkNIT1IiLHMuY291bnQoYSkpCnM9cy5yZXBsYWNlKGEsIGErIlxuICAgICAgICAgICAgICAgICAgICAgICAgaHBhcmFtcy5uX21lcmdlID0gMTsiKQpvcGVuKHAsInciKS53cml0ZShzKQpwcmludCgiTk1FUkdFUEFUQ0ggT0siKQo= | base64 -d | python3 -",
        # DEBUG probe: conv output + transformer output token counts -> /tmp/wenc.txt
        "echo CnA9Ii9sbGFtYS5jcHAvdG9vbHMvbXRtZC9tb2RlbHMvd2hpc3Blci1lbmMuY3BwIgpzPW9wZW4ocCkucmVhZCgpCmlmICIjaW5jbHVkZSA8ZnN0cmVhbT4iIG5vdCBpbiBzOgogICAgcz0iI2luY2x1ZGUgPGZzdHJlYW0+XG4iK3MKYT0nY2IoaW5wLCAiYWZ0ZXJfY29udjFkIiwgLTEpOycKYXNzZXJ0IHMuY291bnQoYSk9PTEsICgiQ09OViIscy5jb3VudChhKSkKcz1zLnJlcGxhY2UoYSwgYSsnIHsgc3RkOjpvZnN0cmVhbSB3ZigiL3RtcC93ZW5jLnR4dCIsc3RkOjppb3M6OmFwcCk7IHdmIDw8ICJhZnRlcl9jb252MWQgbmUwPSIgPDwgaW5wLT5uZVswXSA8PCAiIG5lMT0iIDw8IGlucC0+bmVbMV0gPDwgc3RkOjplbmRsOyB9JykKYTI9J2NiKGN1ciwgImFmdGVyX3RyYW5zZm9ybWVyIiwgLTEpOycKYXNzZXJ0IHMuY291bnQoYTIpPT0xLCAoIlRSIixzLmNvdW50KGEyKSkKcz1zLnJlcGxhY2UoYTIsIGEyKycgeyBzdGQ6Om9mc3RyZWFtIHdmKCIvdG1wL3dlbmMudHh0IixzdGQ6Omlvczo6YXBwKTsgd2YgPDwgImFmdGVyX3RyYW5zZm9ybWVyIG5lMT0iIDw8IGN1ci0+bmVbMV0gPDwgc3RkOjplbmRsOyB9JykKb3BlbihwLCJ3Iikud3JpdGUocykKcHJpbnQoIkNPTlZQUk9CRSBPSyIpCg== | base64 -d | python3 -",
        "cmake -B /llama.cpp/build -S /llama.cpp -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release",
        "cmake --build /llama.cpp/build --target llama-cli llama-completion llama-quantize llama-mtmd-cli -j 8",
    )
    .add_local_dir(".", "/root/moe-lab", ignore=_IGNORE)
)

# lmms-eval (lmms-lab harness) on top of the VLM deps; re-pin transformers last so SigLIP2 stays supported
lmms_image = (
    _vlm_deps
    .apt_install("git")                                    # lmms-eval calls `git describe` for a run hash
    .pip_install("lmms-eval==0.3.0", "sqlitedict", "pytablewriter", "sacrebleu")
    .pip_install("transformers>=4.50,<5")
    .add_local_dir(".", "/root/moe-lab", ignore=_IGNORE)
)

app = modal.App("moe-vlm", image=vlm_image)
data_vol = modal.Volume.from_name("llava-data", create_if_missing=True)   # LLaVA-Pretrain images + json
runs_vol = modal.Volume.from_name("fineweb10B")                           # holds /runs/500M_ctx2048/model.pt
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)     # SigLIP2 weights cache
HF = modal.Secret.from_name("huggingface")

BACKBONE = "/data/runs/500M_ctx2048/model.pt"


@app.function(image=vlm_image, volumes={"/llava": data_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def download():
    """Download LLaVA-Pretrain (LAION-CC-SBU-558K): captions json + images.zip (~24GB), kept as ONE file
    on the volume and STREAMED (random-access by name) at train time — no extracting 558K files."""
    import os, zipfile
    from huggingface_hub import hf_hub_download
    repo = "liuhaotian/LLaVA-Pretrain"
    os.makedirs("/llava", exist_ok=True)
    if not os.path.exists("/llava/blip_laion_cc_sbu_558k.json"):
        hf_hub_download(repo, "blip_laion_cc_sbu_558k.json", repo_type="dataset", local_dir="/llava")
        data_vol.commit()
        print("got captions json", flush=True)
    if not os.path.exists("/llava/images.zip"):
        print("downloading images.zip (~24GB) -> volume (no unzip)...", flush=True)
        hf_hub_download(repo, "images.zip", repo_type="dataset", local_dir="/llava")
        data_vol.commit()
    # sanity: open the zip and confirm the first json image is readable by name
    import json
    data = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    with zipfile.ZipFile("/llava/images.zip") as z:
        names = z.namelist()
        first = data[0]["image"]
        ok = first in z.NameToInfo
    sz = os.path.getsize("/llava/images.zip") / 1e9
    print(f"ready: images.zip ({sz:.1f}GB, {len(names)} entries) + {len(data)} captions | "
          f"first image '{first}' in zip: {ok}", flush=True)


@app.function(image=vlm_image, volumes={"/llava": data_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def download_sft():
    """Stage-2 data: LLaVA-Instruct-150K (the GPT-4 visual-instruction set) + COCO train2017 images
    (~18GB, streamed from the zip like stage 1). Single most effective subset of the 665K mix."""
    import os, json, zipfile, urllib.request
    from huggingface_hub import hf_hub_download
    if not os.path.exists("/llava/llava_instruct_150k.json"):
        hf_hub_download("liuhaotian/LLaVA-Instruct-150K", "llava_instruct_150k.json",
                        repo_type="dataset", local_dir="/llava")
        data_vol.commit()
        print("got llava_instruct_150k.json", flush=True)
    if not os.path.exists("/llava/train2017.zip"):
        print("downloading COCO train2017 (~18GB)...", flush=True)
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/train2017.zip", "/llava/train2017.zip")
        data_vol.commit()
    data = json.load(open("/llava/llava_instruct_150k.json"))
    with zipfile.ZipFile("/llava/train2017.zip") as z:
        names = z.NameToInfo
    hit = sum(1 for ex in data[:300] if f"train2017/{ex['image']}" in names)
    sz = os.path.getsize("/llava/train2017.zip") / 1e9
    print(f"ready: {len(data)} instructions | COCO train2017.zip {sz:.1f}GB ({len(names)} entries) | "
          f"hit-rate first 300: {hit}/300", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def smoke():
    """End-to-end vision path on a synthetic image: SigLIP2 -> project -> splice -> 500M_ctx2048 -> loss/grad."""
    import os, sys, numpy as np, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN, IGNORE_INDEX
    from generate import load_model

    dev = torch.device("cuda")
    torch.manual_seed(0)
    enc = SiglipVision(device=dev)
    img = Image.fromarray((np.random.rand(384, 384, 3) * 255).astype("uint8"))
    feats = enc.encode([img])                                   # (1, N, hidden)
    print(f"SigLIP2 hidden={enc.hidden}  features={tuple(feats.shape)}", flush=True)

    llm, cfg, vloss, step = load_model(BACKBONE, dev)
    print(f"backbone {BACKBONE}: d{cfg.d_model} L{cfg.n_layers} val={vloss} step={step}", flush=True)
    llm.train()
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)

    enc_tok = tiktoken.get_encoding("gpt2")
    cap = enc_tok.encode_ordinary("a photo of a cat sitting on a chair")
    ids = torch.tensor([[IMAGE_TOKEN] + cap], device=dev)       # <image> then caption
    tgt = torch.tensor([[IGNORE_INDEX] + cap], device=dev)      # predict caption only
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = vlm(ids, image_features=feats, targets=tgt)
    loss.backward()
    g = vlm.mm_projector.net[0].weight.grad
    merged_len = feats.shape[1] + len(cap)
    print(f"merged seq len={merged_len}  loss={loss.item():.4f}  ce={parts['ce'].item():.4f}", flush=True)
    print(f"projector grad finite={bool(torch.isfinite(g).all())}  "
          f"backbone frozen? (grad on embed={llm.embed.weight.grad is not None})", flush=True)
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    print("VISION SMOKE OK", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def audio_smoke():
    """End-to-end audio path on synthetic audio: CLAP -> project -> splice <audio> -> 500M backbone -> loss/grad."""
    import os, sys, numpy as np, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from audio import ClapAudio, CLAP_SR
    from multimodal import MoEVLM, AUDIO_TOKEN
    from generate import load_model

    dev = torch.device("cuda")
    torch.manual_seed(0)
    enc = ClapAudio(device=dev)
    wav = (np.random.randn(CLAP_SR * 5).astype("float32")) * 0.1     # 5s of synthetic mono audio
    feats = enc.encode([wav])                                        # (1, T, hidden)
    print(f"CLAP hidden={enc.hidden}  features={tuple(feats.shape)}", flush=True)

    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    print(f"backbone {BACKBONE}: d{cfg.d_model} val={vloss}", flush=True)
    llm.train()
    vlm = MoEVLM(llm, vision_dim=1152, audio_dim=enc.hidden).to(dev)
    tok = tiktoken.get_encoding("gpt2")
    cap = tok.encode_ordinary("the sound of rain falling")
    logical = [AUDIO_TOKEN] + cap + [50256]                          # [<audio>] caption <eot>
    ids = torch.tensor([logical[:-1]], device=dev)
    tgt = torch.tensor([logical[1:]], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = vlm(ids, audio_features=feats, targets=tgt)
    loss.backward()
    g = vlm.audio_projector.net[0].weight.grad
    print(f"merged len={feats.shape[1] + len(cap)}  loss={loss.item():.4f}  "
          f"audio-projector grad finite={bool(torch.isfinite(g).all())}", flush=True)
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    print("AUDIO SMOKE OK", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def train_audio_stage1(max_steps: int = 1200, micro: int = 16, lr: float = 1e-3, warmup: int = 80,
                       save_name: str = "500M_vlm_audio_stage1", repo: str = "CLAPv2/Clotho", log_every: int = 20):
    """Audio stage-1 alignment: train ONLY the audio projector on Clotho; CLAP + LLM frozen. Single H100."""
    import os, sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from torch.utils.data import DataLoader
    from audio import ClapAudio
    from multimodal import MoEVLM
    from generate import load_model
    from vlm_audio_data import ClothoAudio, audio_collate

    dev = torch.device("cuda")
    torch.manual_seed(0); torch.set_float32_matmul_precision("high")
    enc = ClapAudio(device=dev)
    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, audio_dim=enc.hidden).to(dev)
    vlm.set_llm_trainable(False); llm.set_bias_update_rate(0.0)
    n_proj = sum(p.numel() for p in vlm.audio_projector.parameters())
    print(f"backbone d{cfg.d_model} val={vloss:.3f} | CLAP hidden={enc.hidden} | audio-projector={n_proj/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(vlm.audio_projector.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    ds = ClothoAudio(repo)
    dl = DataLoader(ds, batch_size=micro, shuffle=True, num_workers=4, collate_fn=audio_collate,
                    drop_last=True, persistent_workers=True)
    vlm.train()
    step, t0, run, last, done = 0, time.time(), 0.0, float("nan"), False
    while not done:
        for wavs, ids, tgt in dl:
            ids, tgt = ids.to(dev), tgt.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = enc.encode(wavs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = vlm(ids, audio_features=feats, targets=tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.audio_projector.parameters(), 1.0)
            opt.step()
            last = loss.item(); run += last
            if step % log_every == 0:
                print(f"step {step:5d} | loss {last:.4f} | avg {run/(step+1):.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e} | {(time.time()-t0)/(step+1)*1000:.0f}ms/step", flush=True)
            step += 1
            if step >= max_steps:
                done = True; break

    out = f"/data/runs/{save_name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"audio_projector": vlm.audio_projector.state_dict(), "audio_dim": enc.hidden,
                "backbone": BACKBONE, "steps": step}, f"{out}/audio_projector.pt")
    runs_vol.commit()
    print(f"saved audio projector -> {out}/audio_projector.pt (final {last:.4f})", flush=True)
    return {"final_loss": last, "steps": step}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def caption_audio(audio_run: str = "500M_vlm_audio_stage1", n: int = 8, max_new: int = 32,
                  repo: str = "CLAPv2/Clotho"):
    """Greedy-caption a few Clotho clips with the audio VLM; print predicted vs ground-truth."""
    import os, sys, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from audio import ClapAudio
    from multimodal import MoEVLM, AUDIO_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from vlm_audio_data import ClothoAudio

    dev = torch.device("cuda")
    enc = ClapAudio(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, audio_dim=enc.hidden).to(dev)
    ck = torch.load(f"/data/runs/{audio_run}/audio_projector.pt", map_location=dev, weights_only=False)
    vlm.audio_projector.load_state_dict(ck["audio_projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    ds = ClothoAudio(repo)

    def _banned(prev, n=3):
        if len(prev) < n:
            return []
        seen = {}
        for j in range(len(prev) - n + 1):
            seen.setdefault(tuple(prev[j:j + n - 1]), []).append(prev[j + n - 1])
        return seen.get(tuple(prev[-(n - 1):]), [])

    @torch.no_grad()
    def gen(wav):
        feats = enc.encode([wav])
        ids = torch.tensor([[AUDIO_TOKEN]], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, audio_features=feats)
            outs = []
            for _ in range(max_new):
                logits, _ = vlm.llm(inputs_embeds=cur)
                lg = logits[:, -1, :].float(); lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for bnd in _banned(outs, 3):                  # no-repeat-3gram (kills phrase loops)
                    lg[0, bnd] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs)

    for k in range(n):
        i = (k * 619) % len(ds)
        wav, gt = ds.raw(i)
        print(f"\n[{i}] GT:   {str(gt)[:90]}\n     PRED: {gen(wav).strip()}", flush=True)
    print("\nAUDIO CAPTION DONE", flush=True)


@app.function(image=vlm_image, gpu="H100:8", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_stage1(max_steps: int = 1500, micro: int = 32, lr: float = 1e-3, save_name: str = "500M_vlm_stage1"):
    """Stage 1 (alignment) on 8x H100 via torchrun: projector only; LLM + SigLIP2 frozen."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "vlm_stage1.py",
           "--backbone", BACKBONE, "--json", "/llava/blip_laion_cc_sbu_558k.json",
           "--zip", "/llava/images.zip", "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=vlm_image, gpu="H100:8", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_stage2(max_steps: int = 1200, micro: int = 16, lr: float = 2e-5, save_name: str = "500M_vlm_stage2",
                 stage1_run: str = "500M_vlm_stage1"):
    """Stage 2 (SFT) on 8x H100 via torchrun: instruction-tune projector + LLM on LLaVA-Instruct-150K."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "vlm_stage2.py",
           "--backbone", BACKBONE, "--stage1", f"/data/runs/{stage1_run}/projector.pt",
           "--json", "/llava/llava_instruct_150k.json", "--zip", "/llava/train2017.zip", "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def caption(stage1_run: str = "500M_vlm_stage1", n: int = 8, max_new: int = 32, prompt: str = "",
            stage2_run: str = ""):
    """Greedy-caption a few real LAION images with the stage-1 VLM; print predicted vs ground-truth."""
    import os, sys, io, json, zipfile, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    enc = SiglipVision(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    if stage2_run:
        ck = torch.load(f"/data/runs/{stage2_run}/model.pt", map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(ck["model"])              # stage-2 finetuned LLM
        vlm.mm_projector.load_state_dict(ck["projector"])
        print(f"loaded stage-2 VLM from {stage2_run}", flush=True)
    else:
        ck = torch.load(f"/data/runs/{stage1_run}/projector.pt", map_location=dev, weights_only=False)
        vlm.mm_projector.load_state_dict(ck["projector"])
        print(f"loaded stage-1 projector (steps={ck.get('steps')})", flush=True)
    vlm.eval()

    tok = tiktoken.get_encoding("gpt2")
    data = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    z = zipfile.ZipFile("/llava/images.zip")
    if stage2_run:                                   # stage-2 expects the chat format it was trained on
        q = prompt or "Describe this image in detail."
        pre = tok.encode_ordinary(f"USER: {q}\nASSISTANT:")
        print(f"prompt: USER: {q}", flush=True)
    else:
        pre = tok.encode_ordinary(prompt) if prompt else []

    def _banned_ngram(prev, n=3):
        if len(prev) < n:
            return []
        seen = {}
        for j in range(len(prev) - n + 1):
            seen.setdefault(tuple(prev[j:j + n - 1]), []).append(prev[j + n - 1])
        return seen.get(tuple(prev[-(n - 1):]), [])

    @torch.no_grad()
    def gen(image, rep_pen=1.3):
        feats = enc.encode([image])
        ids = torch.tensor([[IMAGE_TOKEN] + pre], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, image_features=feats)
            outs = []
            for _ in range(max_new):
                logits, _ = vlm.llm(inputs_embeds=cur)
                lg = logits[:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                # repetition penalty (CTRL-style) on already-generated tokens
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev)
                    v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / rep_pen, v * rep_pen)
                for b in _banned_ngram(outs, 3):       # block repeating any 3-gram
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                e = vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)
                cur = torch.cat([cur, e], dim=1)
        return tok.decode(outs)

    for k in range(n):
        i = (k * 69779) % len(data)                    # deterministic spread across the set
        ex = data[i]
        img = Image.open(io.BytesIO(z.read(ex["image"]))).convert("RGB")
        gt = ex["conversations"][1]["value"].strip().replace("\n", " ")[:90]
        print(f"\n[{ex['image']}]\n   GT:   {gt}\n   PRED: {gen(img).strip()}", flush=True)
    print("\nCAPTION DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=60 * 60, secrets=[HF])
def ocr_quality(joint_run: str = "500M_vlm_joint10", ckpt: str = "ckpt_2000.pt", n: int = 4, max_new: int = 256,
                backbone: str = "", vision_id: str = ""):
    """In-distribution OCR check: run the model on real ocr_4 images and print PREDICTION vs GROUND-TRUTH.
    Measures how well dense rendered-text reading actually works (the ocr_4 training distribution)."""
    import os, sys, io, json, tarfile, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from PIL import Image
    import tiktoken
    from huggingface_hub import hf_hub_download
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    enc = SiglipVision(model_id=vision_id, device=dev) if vision_id else SiglipVision(device=dev)
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt" if backbone else BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    ck = torch.load(f"/data/runs/{joint_run}/{ckpt}", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(ck["model"]); vlm.mm_projector.load_state_dict(ck["projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")

    jl = hf_hub_download("nvidia/Llama-Nemotron-VLM-Dataset-v1", "ocr_4.jsonl", repo_type="dataset")
    meta = {}
    with open(jl, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            meta[ex["image"].rsplit("/", 1)[-1].rsplit(".", 1)[0]] = ex["conversations"]
    tar = hf_hub_download("nvidia/Llama-Nemotron-VLM-Dataset-v1", "ocr_4_images/shard_000000.tar", repo_type="dataset")

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen(image, q):
        pre = tok.encode_ordinary(f"USER: {q}\nASSISTANT:")
        outs = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = enc.encode([image])
            cur, _ = vlm.build_inputs_embeds(torch.tensor([[IMAGE_TOKEN] + pre], device=dev), image_features=feats)
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs)

    print(f"=== OCR QUALITY {joint_run}/{ckpt} on ocr_4 ===", flush=True)
    got = 0
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            conv = meta.get(base)
            if conv is None:
                continue
            try:
                img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            except Exception:
                continue
            q = (conv[0]["value"] or "").replace("<image>", "").strip()
            gt = (conv[1]["value"] or "").strip().replace("\n", " ")
            pred = gen(img, q).strip().replace("\n", " ")
            print(f"\n--- sample {got} (img {img.size}) ---", flush=True)
            print(f"  GT  : {gt[:240]}", flush=True)
            print(f"  PRED: {pred[:240]}", flush=True)
            got += 1
            if got >= n:
                break
    print("\nOCR QUALITY DONE", flush=True)
    return {"run": joint_run, "ckpt": ckpt, "samples": got}


@app.function(image=vlm_image, volumes={"/cache/hf": hf_cache}, timeout=60 * 60, secrets=[HF])
def ocr_smoke(n: int = 3, shards: int = 1):
    """Validate the ocr_4 OCR data path (basename match + decode + label) BEFORE the big run. Pre-stages the
    jsonl + `shards` tar(s) to the HF cache (Xet off), then iterates StreamOcr and prints image size, token
    shapes, and a decoded snippet of the OCR target. CPU-only, cheap. The staged tars persist for joint10."""
    import os, sys, time
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import tiktoken
    from huggingface_hub import hf_hub_download
    files = ["ocr_4.jsonl"] + [f"ocr_4_images/shard_{i:06d}.tar" for i in range(shards)]
    for path in files:
        for attempt in range(8):
            try:
                hf_hub_download("nvidia/Llama-Nemotron-VLM-Dataset-v1", path, repo_type="dataset"); break
            except Exception as e:
                print(f"stage {path} retry {attempt}: {str(e)[:100]}", flush=True); time.sleep(10)
    from stream_data import StreamOcr
    tok = tiktoken.get_encoding("gpt2")
    ds = StreamOcr(n_shards=shards, rank=0, world=1)
    it = iter(ds); t0 = time.time(); got = 0
    for img, ids, tgt in it:
        labeled = [int(x) for x in tgt.tolist() if x != -1]
        print(f"\n=== ocr sample {got} === img={img.size} ids={tuple(ids.shape)} tgt_labeled={len(labeled)}", flush=True)
        print("  TARGET:", tok.decode(labeled)[:300].replace(chr(10), " "), flush=True)
        got += 1
        if got >= n:
            break
    print(f"\nOCR SMOKE OK: yielded {got} samples in {time.time()-t0:.1f}s", flush=True)
    return {"yielded": got}


@app.function(image=vlm_image, volumes={"/data": runs_vol}, timeout=10 * 60, memory=16384)
def export_projectors(joint_run: str = "500M_vlm_joint12"):
    """Dump the trained projectors (vision mm_projector 1152->768, speech_projector 1536->768) as
    safetensors so the pure-Rust candle app can apply them after candle's SigLIP/Whisper encoders.
    Keys: net.0.{weight,bias} (Linear), net.2.{weight,bias} (Linear); GELU between."""
    import os, torch
    from safetensors.torch import save_file
    exp = f"/data/runs/{joint_run}/export/proj"; os.makedirs(exp, exist_ok=True)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location="cpu", weights_only=False)
    out = {}
    for name, key in [("vision_projector", "projector"), ("speech_projector", "speech_projector")]:
        if key not in jk or jk[key] is None:
            print(f"!! {key} missing from model.pt", flush=True); continue
        sd = {k: v.float().contiguous() for k, v in jk[key].items()}
        save_file(sd, f"{exp}/{name}.safetensors")
        shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in sd.items())
        print(f"wrote {name}.safetensors: {shapes}", flush=True)
        out[name] = {k: list(v.shape) for k, v in sd.items()}
    runs_vol.commit()
    return out


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def export_onnx(joint_run: str = "500M_vlm_joint12",
                vision_id: str = "google/siglip2-so400m-patch16-512",
                speech_only: bool = False):
    """Export the FROZEN encoders + joint12 projectors to ONNX so the desktop app can compute image/speech
    embeddings locally (via `ort`). vision.onnx: pixel(1,3,512,512)->(1,1024,768); speech.onnx: mel(1,80,3000)->
    (1,750,768). Eager attention for clean export. Files -> export/onnx/ on the volume."""
    import os, sys, torch
    import torch.nn as nn
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from transformers import AutoModel, WhisperModel
    from multimodal import Projector
    exp = f"/data/runs/{joint_run}/export/onnx"; os.makedirs(exp, exist_ok=True)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location="cpu", weights_only=False)
    D = 768

    # ---- vision: SigLIP2 vision tower + mm_projector ----
    if not speech_only:
        vfull = AutoModel.from_pretrained(vision_id, torch_dtype=torch.float32, attn_implementation="eager")
        vmodel = vfull.vision_model.float().eval()
        vproj = Projector(vmodel.config.hidden_size, D)
        vproj.load_state_dict(jk["projector"]); vproj.float().eval()

        class VWrap(nn.Module):
            def __init__(self):
                super().__init__(); self.v = vmodel; self.p = vproj
            def forward(self, pixel_values):
                h = self.v(pixel_values=pixel_values).last_hidden_state
                return self.p(h)

        with torch.no_grad():
            torch.onnx.export(VWrap().eval(), (torch.zeros(1, 3, 512, 512),), f"{exp}/vision.onnx",
                              input_names=["pixel_values"], output_names=["embeds"],
                              opset_version=17, do_constant_folding=True)
        print(f"wrote {exp}/vision.onnx ({os.path.getsize(f'{exp}/vision.onnx')/1e6:.0f} MB)", flush=True)

    # ---- speech: Whisper-small encoder + stack2 + speech_projector ----
    wfull = WhisperModel.from_pretrained("openai/whisper-small", torch_dtype=torch.float32,
                                         attn_implementation="eager")
    wenc = wfull.encoder.float().eval()
    sproj = Projector(wfull.config.d_model * 2, D)  # stack=2 -> 1536
    sproj.load_state_dict(jk["speech_projector"]); sproj.float().eval()

    class SWrap(nn.Module):
        def __init__(self):
            super().__init__(); self.e = wenc; self.p = sproj
        def forward(self, mel):
            h = self.e(mel).last_hidden_state  # (B,1500,768)
            b, t, c = h.shape
            h = h.reshape(b, t // 2, c * 2)    # stack adjacent frames -> (B,750,1536)
            return self.p(h)

    with torch.no_grad():
        torch.onnx.export(SWrap().eval(), (torch.zeros(1, 80, 3000),), f"{exp}/speech.onnx",
                          input_names=["mel"], output_names=["embeds"],
                          opset_version=17, do_constant_folding=True)
    print(f"wrote {exp}/speech.onnx ({os.path.getsize(f'{exp}/speech.onnx')/1e6:.0f} MB)", flush=True)

    # ---- dump the exact Whisper mel filterbank + feature config so the Rust app matches bit-for-bit ----
    import numpy as np
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    mf = np.asarray(fe.mel_filters, dtype=np.float32)        # (201, 80) freq x mel
    mf80 = np.ascontiguousarray(mf.T if mf.shape[0] != 80 else mf, dtype=np.float32)  # (80, 201)
    mf80.tofile(f"{exp}/mel_filters_80x201.bin")
    print(f"wrote mel_filters_80x201.bin shape={mf80.shape} | n_fft={fe.n_fft} hop={fe.hop_length} "
          f"chunk={fe.chunk_length}s sr={fe.sampling_rate} nmels={fe.feature_size}", flush=True)
    runs_vol.commit()
    out = {"speech_mb": round(os.path.getsize(f"{exp}/speech.onnx") / 1e6)}
    if not speech_only:
        out["vision_mb"] = round(os.path.getsize(f"{exp}/vision.onnx") / 1e6)
    return out


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def dump_embeds(joint_run: str = "500M_vlm_joint12",
                img_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
                audio_url: str = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav",
                vision_id: str = "google/siglip2-so400m-patch16-512",
                do_image: bool = True, do_speech: bool = True):
    """Run the FROZEN encoders + joint12 projectors and dump the PROJECTED (N x 768) embeddings to .bin files
    (raw f32) for the Rust engine's --image/--speech inputs_embeds. Uses PyTorch encoders (correct 1024 image /
    750 speech token counts) — sidesteps the clip.cpp audio downsample bug entirely."""
    import os, sys, urllib.request, torch
    import numpy as np
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    from vision import SiglipVision
    from speech import WhisperSpeech
    from multimodal import Projector
    dev = torch.device("cuda")
    exp = f"/data/runs/{joint_run}/export"; os.makedirs(exp, exist_ok=True)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location="cpu", weights_only=False)
    D = 768
    res = {}
    if do_image:
        vis = SiglipVision(model_id=vision_id, device=dev, dtype=torch.float32)
        proj = Projector(vis.hidden, D); proj.load_state_dict(jk["projector"]); proj.to(dev).float().eval()
        urllib.request.urlretrieve(img_url, "/tmp/img.jpg")
        img = Image.open("/tmp/img.jpg").convert("RGB")
        with torch.no_grad():
            feats = vis.encode([img]).float()                  # (1, N, 1152)
            emb = proj(feats)[0].cpu().numpy().astype(np.float32)  # (N, 768)
        emb.tofile(f"{exp}/image_embeds.bin")
        print(f"image embeds {emb.shape} -> {exp}/image_embeds.bin", flush=True); res["image"] = list(emb.shape)
    if do_speech:
        import soundfile as sf
        spk = WhisperSpeech(device=dev, dtype=torch.float32)
        sproj = Projector(spk.hidden, D); sproj.load_state_dict(jk["speech_projector"]); sproj.to(dev).float().eval()
        urllib.request.urlretrieve(audio_url, "/tmp/a.wav")
        wav, sr = sf.read("/tmp/a.wav")
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(1)
        with torch.no_grad():
            feats = spk.encode([wav.astype("float32")], sr=sr).float()  # (1, 750, 1536)
            emb = sproj(feats)[0].cpu().numpy().astype(np.float32)
        emb.tofile(f"{exp}/speech_embeds.bin")
        print(f"speech embeds {emb.shape} -> {exp}/speech_embeds.bin", flush=True); res["speech"] = list(emb.shape)
    runs_vol.commit()
    return res


@app.function(image=litert_image, cpu=8.0, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=60 * 60, secrets=[HF])
def mtmd_vision(joint_run: str = "500M_vlm_joint12",
                img_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
                prompt: str = "Describe this image in detail.", n: int = 64, push_repo: str = "moe-omni-500m"):
    """VISION e2e: convert SigLIP2+mm_projector -> mmproj GGUF (proj_type=phi4), then run llama-mtmd-cli with our
    LLM GGUF + the mmproj on a real image -> see if it describes the image. The hard multimodal-splice test."""
    import os, sys, subprocess, urllib.request
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    exp = f"/data/runs/{joint_run}/export"; os.makedirs(exp, exist_ok=True)
    llm = f"{exp}/joint12-bailingmoe2.gguf"; mmproj = f"{exp}/mmproj-joint12.gguf"
    ckpt = f"/data/runs/{joint_run}/model.pt"
    subprocess.run([sys.executable, "to_gguf.py", "--ckpt", ckpt, "--out", llm], check=True)   # +chat template
    subprocess.run([sys.executable, "to_mmproj.py", "--ckpt", ckpt, "--out", mmproj], check=True)
    img = "/tmp/test.jpg"; urllib.request.urlretrieve(img_url, img)
    cli = subprocess.run(["bash", "-lc", "find /llama.cpp -name llama-mtmd-cli | head -1"],
                         capture_output=True, text=True).stdout.strip()
    full = f"USER: {prompt}\nASSISTANT:"     # marker auto-prepended by mtmd -> [IMAGE]USER: q\nASSISTANT:
    print(f"\n=== RUN: llama-mtmd-cli -m LLM --mmproj mmproj --image (cats) -p '{full}' ===", flush=True)
    r = subprocess.run([cli, "-m", llm, "--mmproj", mmproj, "--image", img, "-p", full, "--jinja",
                        "-n", str(n), "--temp", "0", "-t", "8"], capture_output=True, text=True, timeout=600)
    print("=== STDOUT ===\n" + r.stdout[-2500:], flush=True)
    print("=== STDERR (tail) ===\n" + r.stderr[-2500:], flush=True)
    ok = bool(r.stdout.strip()) and r.returncode == 0
    if ok:
        try:
            from huggingface_hub import HfApi
            api = HfApi(); repo_id = f"{api.whoami()['name']}/{push_repo}"
            api.upload_file(path_or_fileobj=mmproj, path_in_repo="export/mmproj-joint12.gguf", repo_id=repo_id)
            print(f"uploaded mmproj -> https://huggingface.co/{repo_id}/tree/main/export", flush=True)
        except Exception as e:
            print(f"(upload skipped: {str(e)[:120]})", flush=True)
    return {"ran": ok}


@app.function(image=litert_image, cpu=8.0, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def audio_mel_debug(joint_run: str = "500M_vlm_joint12",
                    audio_url: str = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"):
    """LOCALIZE the audio gap: run clip on jfk.wav (the patched build dumps clip's mel -> /tmp/clip_mel.bin),
    then compute HF WhisperFeatureExtractor mel for the SAME 30s wav and compare. mel-match => bug is downstream
    (encoder weights); mel-differ => clip's mel preprocessing is the bug."""
    import os, sys, subprocess, urllib.request, struct
    import numpy as np
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import soundfile as sf
    exp = f"/data/runs/{joint_run}/export"; ckpt = f"/data/runs/{joint_run}/model.pt"
    llm = f"{exp}/joint12-bailingmoe2.gguf"; aproj = f"{exp}/aproj-joint12.gguf"
    if not os.path.exists(llm):
        subprocess.run([sys.executable, "to_gguf.py", "--ckpt", ckpt, "--out", llm], check=True)
    if not os.path.exists(aproj):
        subprocess.run([sys.executable, "to_aproj.py", "--ckpt", ckpt, "--out", aproj], check=True)
    urllib.request.urlretrieve(audio_url, "/tmp/j.wav")
    y0, sr = sf.read("/tmp/j.wav")
    if y0.ndim > 1:
        y0 = y0.mean(1)
    y0 = y0.astype("float32")
    y30 = np.concatenate([y0, np.zeros(30 * sr - len(y0), dtype=y0.dtype)]) if len(y0) < 30 * sr else y0[:30 * sr]
    cli = subprocess.run(["bash", "-lc", "find /llama.cpp -name llama-mtmd-cli | head -1"],
                         capture_output=True, text=True).stdout.strip()
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    for tag, y in [("RAW", y0), ("PAD30", y30)]:
        sf.write("/tmp/jx.wav", y, sr)
        if os.path.exists("/tmp/clip_mel.bin"):
            os.remove("/tmp/clip_mel.bin")
        subprocess.run(["bash", "-lc", f"timeout -s KILL 300 {cli} -m {llm} --mmproj {aproj} --audio /tmp/jx.wav "
                        f"-p USER:hi --jinja -n 1 --temp 0 -t 8 --no-warmup >/tmp/m.out 2>/tmp/m.err"],
                       stdin=subprocess.DEVNULL)
        raw = open("/tmp/clip_mel.bin", "rb").read()
        n_mel, n_len = struct.unpack("ii", raw[:8])
        clip_mel = np.frombuffer(raw[8:], dtype=np.float32).reshape(n_mel, n_len)
        hf = fe(y, sampling_rate=sr, return_tensors="np").input_features[0]      # (80, 3000)
        L = min(clip_mel.shape[1], hf.shape[1]); cm = clip_mel[:, :L]; hm = hf[:, :L]
        print(f"\n[{tag}] in={len(y)/sr:.1f}s  clip_mel={clip_mel.shape}  hf={hf.shape}  "
              f"MSE(first{L})={((cm - hm) ** 2).mean():.5f}  corr={np.corrcoef(cm.flatten(), hm.flatten())[0, 1]:.4f}",
              flush=True)
        # how many audio tokens did mtmd make? (grep the cli stderr for the embeddings/tokens count)
        et = subprocess.run(["bash", "-lc", "grep -iE 'audio|embd|token|n_pos|chunk|encode' /tmp/m.err | tail -8"],
                            capture_output=True, text=True)
        print(et.stdout, flush=True)


@app.function(image=litert_image, cpu=8.0, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def audio_embd_debug(joint_run: str = "500M_vlm_joint12",
                     audio_url: str = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"):
    """DEFINITIVE bisection: dump clip's PROJECTED audio embeddings (/tmp/clip_embd.bin via the patched build),
    then compute PyTorch WhisperSpeech+speech_projector embeddings for the SAME 30s clip (CPU) and compare
    per-token. match => engine path correct (bug downstream); differ => encoder/projector port bug (localize)."""
    import os, sys, subprocess, urllib.request, torch
    import numpy as np
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import soundfile as sf
    exp = f"/data/runs/{joint_run}/export"; ckpt = f"/data/runs/{joint_run}/model.pt"
    llm = f"{exp}/joint12-bailingmoe2.gguf"; aproj = f"{exp}/aproj-joint12.gguf"
    for src, out in [("to_gguf.py", llm), ("to_aproj.py", aproj)]:
        if not os.path.exists(out):
            subprocess.run([sys.executable, src, "--ckpt", ckpt, "--out", out], check=True)
    urllib.request.urlretrieve(audio_url, "/tmp/j.wav")
    y, sr = sf.read("/tmp/j.wav")
    if y.ndim > 1:
        y = y.mean(1)
    y = y.astype("float32")
    y30 = np.concatenate([y, np.zeros(30 * sr - len(y), dtype=y.dtype)]) if len(y) < 30 * sr else y[:30 * sr]
    sf.write("/tmp/j30.wav", y30, sr)
    for fp in ["/tmp/clip_embd.bin", "/tmp/wenc.txt"]:
        if os.path.exists(fp):
            os.remove(fp)
    cli = subprocess.run(["bash", "-lc", "find /llama.cpp -name llama-mtmd-cli | head -1"],
                         capture_output=True, text=True).stdout.strip()
    subprocess.run(["bash", "-lc", f"timeout -s KILL 300 {cli} -m {llm} --mmproj {aproj} --audio /tmp/j30.wav "
                    f"-p USER:hi --jinja -n 1 --temp 0 -t 8 --no-warmup >/tmp/m.out 2>/tmp/m.err"],
                   stdin=subprocess.DEVNULL)
    print("clip whisper-enc sizes:\n" + (open("/tmp/wenc.txt").read() if os.path.exists("/tmp/wenc.txt") else "(none)"),
          flush=True)
    ce = np.frombuffer(open("/tmp/clip_embd.bin", "rb").read(), dtype=np.float32).reshape(-1, 768)
    # PyTorch reference (CPU, fp32)
    from speech import WhisperSpeech, WHISPER_SR
    from multimodal import Projector
    enc = WhisperSpeech(device="cpu", dtype=torch.float32)
    proj = Projector(enc.hidden, 768)
    proj.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["speech_projector"]); proj.eval()
    with torch.no_grad():
        feats = enc.encode([y30], sr=sr)                      # (1, 750, 1536)
        pt = proj(feats)[0].float().numpy()                   # (750, 768)
    print(f"\nclip_embd {ce.shape}  pytorch {pt.shape}", flush=True)
    n = min(len(ce), len(pt)); a = ce[:n]; b = pt[:n]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    print(f"clip  stats mean={a.mean():.4f} std={a.std():.4f} min={a.min():.3f} max={a.max():.3f}", flush=True)
    print(f"torch stats mean={b.mean():.4f} std={b.std():.4f} min={b.min():.3f} max={b.max():.3f}", flush=True)
    print(f"per-token cosine: mean={cos.mean():.4f} min={cos.min():.4f}  MSE={((a-b)**2).mean():.5f}", flush=True)
    print(f"first-token cos={cos[0]:.4f}  mid={cos[n//2]:.4f}  last={cos[-1]:.4f}", flush=True)


@app.function(image=litert_image, cpu=4.0, timeout=10 * 60)
def inspect_convert():
    """Grep llama.cpp's OWN Ultravox/Voxtral mmproj converter for the whisper-encoder tensor mapping — to spot
    any transform (transpose/permute/scale on conv or q/k/v) our to_aproj.py is missing."""
    import subprocess
    # build_vit FULL body: find the def line, dump it, find where ne[1] gets halved (merge/pool/reshape/conv)
    ln = int(subprocess.run(["bash", "-lc", "grep -n 'clip_graph::build_vit' /llama.cpp/tools/mtmd/clip.cpp | head -1 | cut -d: -f1"],
                            capture_output=True, text=True).stdout.strip() or "0")
    print(f"=== build_vit @ {ln} (full body, key ops) ===", flush=True)
    # dump the build_vit EPILOGUE (after the layer loop) where the 1500->750 reduction must be
    r = subprocess.run(["bash", "-lc", f"awk 'NR>={ln+150} && NR<{ln+260} {{print NR\": \"$0}}' /llama.cpp/tools/mtmd/clip.cpp"],
                       capture_output=True, text=True)
    print("=== build_vit epilogue ===\n" + r.stdout[-3800:], flush=True)


@app.function(image=litert_image, cpu=8.0, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=60 * 60, secrets=[HF])
def mtmd_audio(joint_run: str = "500M_vlm_joint12",
               audio_url: str = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav",
               prompt: str = "What is being said in the audio?", n: int = 80, pad30: bool = True,
               push_repo: str = "moe-omni-500m"):
    """AUDIO/SPEECH e2e: convert whisper-small enc + speech_projector -> audio mmproj GGUF (proj_type=voxtral),
    then run llama-mtmd-cli with our LLM GGUF + the audio mmproj on a real speech wav -> see if it responds on
    topic. clip.cpp does wav->mel->whisper->stack itself; mtmd splices at the audio marker (= [SPEECH_TOKEN])."""
    import os, sys, json, subprocess, urllib.request
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    exp = f"/data/runs/{joint_run}/export"; os.makedirs(exp, exist_ok=True)
    llm = f"{exp}/joint12-bailingmoe2.gguf"; aproj = f"{exp}/aproj-joint12.gguf"
    ckpt = f"/data/runs/{joint_run}/model.pt"
    subprocess.run([sys.executable, "to_gguf.py", "--ckpt", ckpt, "--out", llm], check=True)   # +chat template
    subprocess.run([sys.executable, "to_aproj.py", "--ckpt", ckpt, "--out", aproj], check=True)
    wav = "/tmp/test.wav"; urllib.request.urlretrieve(audio_url, wav)
    if pad30:        # our model ALWAYS trained on whisper's 30s-padded input (1500 frames -> 750 tok); if mtmd
        import soundfile as sf; import numpy as np          # encodes only the raw clip it's OOD -> pad to 30s
        y, sr = sf.read(wav)
        if y.ndim > 1:
            y = y.mean(1)
        need = 30 * sr - len(y)
        if need > 0:
            y = np.concatenate([y, np.zeros(need, dtype=y.dtype)])
        wav = "/tmp/test30.wav"; sf.write(wav, y, sr)
        print(f"padded audio to 30s ({len(y)} samples @ {sr}Hz)", flush=True)
    cli = subprocess.run(["bash", "-lc", "find /llama.cpp -name llama-mtmd-cli | head -1"],
                         capture_output=True, text=True).stdout.strip()
    full = f"USER: {prompt}\nASSISTANT:"     # marker auto-prepended by mtmd -> [SPEECH]USER: q\nASSISTANT:
    so, se = f"{exp}/mtmd_audio.out", f"{exp}/mtmd_audio.err"      # tee to volume -> readable even if killed
    print(f"\n=== RUN: llama-mtmd-cli -m LLM --mmproj aproj --audio (jfk.wav) -p '{full}' n={n} ===", flush=True)
    # OS `timeout` = hard SIGKILL bound; stdin=DEVNULL prevents any interactive-turn hang; --no-warmup skips the
    # empty warmup run. Tee both streams to the volume so a function-timeout still leaves the encode timing behind.
    cmd = (f"timeout -s KILL 900 {cli} -m {llm} --mmproj {aproj} --audio {wav} "
           f"-p {json.dumps(full)} --jinja -n {n} --temp 0 -t 8 --no-warmup "
           f"> {so} 2> {se}; echo EXIT=$?")
    tail = subprocess.run(["bash", "-lc", cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1000)
    out = open(so).read() if os.path.exists(so) else ""
    err = open(se).read() if os.path.exists(se) else ""
    print("EXIT:", tail.stdout.strip(), flush=True)
    print("=== STDOUT ===\n" + out[-2500:], flush=True)
    print("=== STDERR (tail) ===\n" + err[-3500:], flush=True)
    ok = bool(out.strip()) and "EXIT=0" in tail.stdout
    if ok:
        try:
            from huggingface_hub import HfApi
            api = HfApi(); repo_id = f"{api.whoami()['name']}/{push_repo}"
            api.upload_file(path_or_fileobj=aproj, path_in_repo="export/aproj-joint12.gguf", repo_id=repo_id)
            print(f"uploaded audio mmproj -> https://huggingface.co/{repo_id}/tree/main/export", flush=True)
        except Exception as e:
            print(f"(upload skipped: {str(e)[:120]})", flush=True)
    return {"ran": ok}


@app.function(image=litert_image, cpu=8.0, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=60 * 60, secrets=[HF])
def llamacpp_quantize(joint_run: str = "500M_vlm_joint12", qtype: str = "Q4_K_M",
                      prompt: str = "The capital of France is", n: int = 48, push_repo: str = "moe-omni-500m"):
    """Quantize the fp32 GGUF -> Q4_K_M (or Q8_0) with llama-quantize, run the quantized model to verify quality,
    report sizes + speed. Completes the efficient CPU/GPU engine for the LLM core."""
    import os, subprocess
    exp = f"/data/runs/{joint_run}/export"
    src = f"{exp}/joint12-bailingmoe2.gguf"
    dst = f"{exp}/joint12-bailingmoe2-{qtype}.gguf"
    quant = subprocess.run(["bash", "-lc", "find /llama.cpp -name llama-quantize | head -1"],
                           capture_output=True, text=True).stdout.strip()
    cli = subprocess.run(["bash", "-lc", "find /llama.cpp -name llama-completion | head -1"],
                         capture_output=True, text=True).stdout.strip()
    print(f"quantize {os.path.getsize(src)/1e9:.2f}GB fp32 -> {qtype} ...", flush=True)
    subprocess.run([quant, src, dst, qtype], check=True)
    print(f"=== {qtype}: {os.path.getsize(dst)/1e6:.0f} MB (was {os.path.getsize(src)/1e9:.2f} GB) ===", flush=True)
    r = subprocess.run([cli, "-m", dst, "-p", prompt, "-n", str(n), "--temp", "0", "-t", "8"],
                       capture_output=True, text=True, timeout=600)
    print("=== STDOUT (quantized) ===\n" + r.stdout[-1500:], flush=True)
    print("=== perf ===\n" + "\n".join(l for l in r.stderr.splitlines() if "tokens per second" in l or "eval time" in l)[-600:], flush=True)
    runs_vol.commit()
    try:
        from huggingface_hub import HfApi
        api = HfApi(); repo_id = f"{api.whoami()['name']}/{push_repo}"
        api.upload_file(path_or_fileobj=dst, path_in_repo=f"export/{os.path.basename(dst)}", repo_id=repo_id)
        print(f"uploaded -> https://huggingface.co/{repo_id}/tree/main/export", flush=True)
    except Exception as e:
        print(f"(upload skipped: {str(e)[:120]})", flush=True)
    return {"qtype": qtype, "mb": round(os.path.getsize(dst) / 1e6)}


@app.function(image=litert_image, cpu=8.0, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=60 * 60, secrets=[HF])
def llamacpp_e2e(joint_run: str = "500M_vlm_joint12", prompt: str = "The capital of France is",
                 n: int = 48, push_repo: str = "moe-omni-500m"):
    """END-TO-END custom-engine test: convert joint12 -> GGUF (arch=bailingmoe2) and RUN it in stock llama.cpp
    (native sparse MoE on CPU). Validates the model loads + generates coherent text, then uploads the GGUF."""
    import os, sys, subprocess
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    out = f"/data/runs/{joint_run}/export/joint12-bailingmoe2.gguf"; os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([sys.executable, "to_gguf.py", "--ckpt", f"/data/runs/{joint_run}/model.pt", "--out", out],
                   check=True)
    cli = subprocess.run(["bash", "-lc", "ls /llama.cpp/build/bin/llama-completion 2>/dev/null || "
                          "find /llama.cpp -name llama-completion -o -name llama-cli | head -1"],
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"\n=== RUN: {cli} -m joint12-bailingmoe2.gguf -p '{prompt}' -n {n} ===", flush=True)
    r = subprocess.run([cli, "-m", out, "-p", prompt, "-n", str(n), "--temp", "0", "-t", "8", "--jinja"],
                       capture_output=True, text=True, timeout=600)
    print("=== STDOUT ===\n" + r.stdout[-2500:], flush=True)
    print("=== STDERR (tail) ===\n" + r.stderr[-2000:], flush=True)
    ok = bool(r.stdout.strip()) and r.returncode == 0
    if ok:
        try:
            from huggingface_hub import HfApi
            api = HfApi(); repo_id = f"{api.whoami()['name']}/{push_repo}"
            api.upload_file(path_or_fileobj=out, path_in_repo="export/joint12-bailingmoe2.gguf", repo_id=repo_id)
            print(f"uploaded -> https://huggingface.co/{repo_id}/tree/main/export", flush=True)
        except Exception as e:
            print(f"(upload skipped: {str(e)[:120]})", flush=True)
    return {"ran": ok}


@app.function(image=vlm_image, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=2 * 60 * 60, secrets=[HF])
def convert_formats(joint_run: str = "500M_vlm_joint12", backbone: str = "500M_ctx8k", do_onnx: bool = True,
                    seq: int = 64, push_repo: str = "moe-omni-500m"):
    """Convert the final joint12 checkpoint to (1) safetensors (all weights flattened + config.json) and
    (2) ONNX of the LLM core (idx->logits, single forward; bmm MoE backend = standard ops). Frozen encoders
    (SigLIP2/CLAP/Whisper) keep their own ONNX; projectors are tiny MLPs. Uploads to HF under export/."""
    import os, sys, json, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from safetensors.torch import save_file
    dev = torch.device("cpu")
    out = f"/data/runs/{joint_run}"; os.makedirs(f"{out}/export", exist_ok=True)
    jk = torch.load(f"{out}/model.pt", map_location=dev, weights_only=False)

    # ---- (1) SAFETENSORS: flatten every weight group under a prefix; config -> json ----
    flat, groups = {}, ["model", "projector", "audio_projector", "speech_projector"]
    for g in groups:
        for k, v in (jk.get(g) or {}).items():
            if isinstance(v, torch.Tensor):
                # .clone() breaks tied-weight memory sharing (lm_head/embed) which save_file rejects
                flat[f"{('llm' if g == 'model' else g)}.{k}"] = v.detach().clone().to(torch.float32).contiguous()
    save_file(flat, f"{out}/export/joint12.safetensors",
              metadata={"format": "pt", "model": "moe-omni-500m-joint12", "groups": ",".join(groups)})
    cfg = jk.get("config", {})
    json.dump({**cfg, "vision_encoder": "google/siglip2-so400m-patch16-512",
               "speech_encoder": "openai/whisper-small", "audio_encoder": "laion/clap-htsat-unfused", "ctx": 8192,
               "note": "weights prefixed llm./mm_projector./audio_projector./speech_projector.; rope_theta carried here"},
              open(f"{out}/export/config.json", "w"), indent=2)
    print(f"safetensors: {len(flat)} tensors -> joint12.safetensors  (+config.json)", flush=True)

    # ---- (2) ONNX: LLM core (idx -> logits). CPU load => bmm backend (no grouped_mm custom op) ----
    onnx_ok = False
    if do_onnx:
        from generate import load_model
        llm, mcfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt", dev)   # cpu -> expert_backend='bmm'
        llm.load_state_dict(jk["model"]); llm.eval()
        for m in llm.modules():
            if hasattr(m, "backend"):
                m.backend = "bmm"

        class LLMWrap(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, input_ids): return s.m(input_ids)[0]
        idx = torch.randint(0, mcfg.vocab_size, (1, seq))
        try:
            with torch.no_grad():
                torch.onnx.export(LLMWrap(llm).eval(), (idx,), f"{out}/export/joint12_llm.onnx",
                                  input_names=["input_ids"], output_names=["logits"],
                                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                                "logits": {0: "batch", 1: "seq"}},
                                  opset_version=17, do_constant_folding=True, dynamo=False)  # legacy TS exporter
            print(f"ONNX: exported joint12_llm.onnx ({os.path.getsize(f'{out}/export/joint12_llm.onnx')/1e6:.0f} MB)",
                  flush=True)
            onnx_ok = True
        except Exception as e:
            print(f"ONNX export FAILED (custom/unsupported op): {type(e).__name__}: {str(e)[:300]}", flush=True)

    runs_vol.commit()
    try:
        from huggingface_hub import HfApi
        api = HfApi(); repo_id = f"{api.whoami()['name']}/{push_repo}"
        for f in ["joint12.safetensors", "config.json"] + (["joint12_llm.onnx"] if onnx_ok else []):
            api.upload_file(path_or_fileobj=f"{out}/export/{f}", path_in_repo=f"export/{f}", repo_id=repo_id)
            print(f"uploaded export/{f}", flush=True)
        print(f"DONE -> https://huggingface.co/{repo_id}/tree/main/export", flush=True)
    except Exception as e:
        print(f"(HF upload skipped: {str(e)[:120]})", flush=True)
    return {"safetensors": True, "onnx": onnx_ok}


@app.function(image=vlm_image, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=3 * 60 * 60, secrets=[HF])
def newdata_smoke(n: int = 2):
    """Validate the 3 new joint12 data paths (smoltalk / mobile-actions / Aria desktop grounding) BEFORE the big
    run: stage each, iterate the streamer, print decoded samples. Aria also checks img-basename matching against
    the screenshots zip (the make-or-break, like ocr_smoke). CPU-only."""
    import os, sys
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import tiktoken
    from huggingface_hub import hf_hub_download
    tok = tiktoken.get_encoding("gpt2")

    def show(tag, ds, k):
        got = 0
        for item in ds:
            if len(item) == 3:
                img, ids, tgt = item; extra = f"img={img.size}"
            else:
                ids, tgt = item; extra = ""
            lab = [int(x) for x in tgt.tolist() if x != -1]
            print(f"\n[{tag} {got}] ids={tuple(ids.shape)} {extra} TARGET: {tok.decode(lab)[:200]}", flush=True)
            got += 1
            if got >= k:
                break
        print(f"{tag}: yielded {got}", flush=True)
        return got

    from stream_data import StreamSmolTalk, StreamMobileActions, StreamAria, aria_preprocess
    hf_hub_download("HuggingFaceTB/smoltalk", "data/all/train-00000-of-00009.parquet", repo_type="dataset")
    show("smoltalk", StreamSmolTalk(files=("data/all/train-00000-of-00009.parquet",)), n)
    hf_hub_download("google/mobile-actions", "dataset.jsonl", repo_type="dataset")
    show("mobile", StreamMobileActions(), n)
    jp = hf_hub_download("Aria-UI/Aria-UI_Data", "desktop/aria_ui_desktop_with_instructions.json", repo_type="dataset")
    zp = hf_hub_download("Aria-UI/Aria-UI_Data", "desktop/screenshots.zip", repo_type="dataset")
    os.makedirs("/data/aria", exist_ok=True)
    nrec = aria_preprocess(jp, "/data/aria/smoke.jsonl", max_samples=3000)
    print(f"\naria_preprocess -> {nrec} grounding samples", flush=True)
    got = show("aria", StreamAria("/data/aria/smoke.jsonl", zp), n)
    print(f"\nNEWDATA SMOKE {'OK' if got else 'FAILED (aria 0 — basename mismatch?)'}", flush=True)
    return {"aria_yielded": got}


@app.function(image=vlm_image, gpu="H100:8", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_joint(max_steps: int = 1600, micro: int = 4, lr: float = 2e-5, save_name: str = "500M_vlm_joint",
                stage2_run: str = "500M_vlm_stage2", audio_run: str = "500M_vlm_audio_stage1",
                speech_run: str = "", with_tools: bool = False, with_agent: bool = False, tools_rep: int = 1,
                init_joint: str = "", stream: bool = False, stream_ocr: int = 0,
                backbone: str = "500M_ctx2048", vision_id: str = "", ocr_max_len: int = 2048,
                stream_smol: int = 0, stream_mobile: int = 0, with_aria: bool = False, aria_max: int = 200000):
    """Joint multimodal SFT on 8x H100 (torchrun): co-train LLM + mm_projector + audio_projector (+ speech_projector
    if speech_run -> 5 paths; + tool-use path if with_tools -> 6 paths: image/video/audio/speech/text/tools)."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "vlm_joint.py",
           "--stage2", f"/data/runs/{stage2_run}/model.pt", "--audio", f"/data/runs/{audio_run}/audio_projector.pt",
           "--json", "/llava/llava_instruct_150k.json", "--zip", "/llava/train2017.zip",
           "--clotho", "CLAPv2/Clotho", "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr),
           "--backbone", f"/data/runs/{backbone}/model.pt", "--ocr_max_len", str(ocr_max_len)]
    if vision_id:
        cmd += ["--vision_id", vision_id]
    if speech_run:
        cmd += ["--speech", f"/data/runs/{speech_run}/speech_projector.pt", "--va", "gpt-omni/VoiceAssistant-400K"]
    if with_tools:
        cmd += ["--tools", "/data/tools/tools_train.jsonl", "--tools_rep", str(tools_rep)]
    if with_agent:
        cmd += ["--text_traj", "/data/tools/nemotron_traj_train.jsonl",
                "--speech_tool", "/data/tools/speech_tool_train.jsonl"]
    if init_joint:
        cmd += ["--init_joint", f"/data/runs/{init_joint}/model.pt"]
    if stream:
        # pre-download the streaming JSONL ONCE here (single process, retries) so the 100s of dataloader
        # workers read the cache via local_files_only -> no concurrent SSL storm against huggingface.co
        import time
        from huggingface_hub import hf_hub_download
        for repo, path in [("nvidia/Nemotron-SFT-Agentic-v2", "data/tool_calling.jsonl"),
                           ("nvidia/Nemotron-SFT-Agentic-v2", "data/search.jsonl"),
                           ("nvidia/Nemotron-SFT-Agentic-v2", "data/interactive_agent.jsonl"),
                           ("nvidia/Llama-Nemotron-VLM-Dataset-v1", "captioning_1.jsonl"),
                           ("nvidia/Llama-Nemotron-VLM-Dataset-v1", "captioning_2.jsonl")]:
            for attempt in range(6):
                try:
                    p = hf_hub_download(repo, path, repo_type="dataset")
                    print(f"predownloaded {path} -> {p}", flush=True)
                    break
                except Exception as e:
                    print(f"predownload {path} retry {attempt}: {str(e)[:120]}", flush=True)
                    time.sleep(8)
        cmd += ["--stream_agentic", "1", "--stream_vlm", "1"]
        if stream_ocr:
            # pre-stage ocr_4 (jsonl + 9 image tar shards, ~33GB) to the HF cache ONCE, Xet disabled, so the
            # dataloader workers read the tars locally (no Xet/SSL storm at train time). Persists on hf-cache vol.
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            ocr_files = ["ocr_4.jsonl"] + [f"ocr_4_images/shard_{i:06d}.tar" for i in range(9)]
            for path in ocr_files:
                for attempt in range(8):
                    try:
                        p = hf_hub_download("nvidia/Llama-Nemotron-VLM-Dataset-v1", path, repo_type="dataset")
                        print(f"predownloaded OCR {path} -> {p}", flush=True)
                        break
                    except Exception as e:
                        print(f"predownload OCR {path} retry {attempt}: {str(e)[:120]}", flush=True)
                        time.sleep(10)
            cmd += ["--stream_ocr", str(stream_ocr)]
    if stream_smol or stream_mobile or with_aria:
        import time, sys as _sys
        _sys.path.insert(0, "/root/moe-lab")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import hf_hub_download

        def _stage(repo, path, tag):
            for attempt in range(8):
                try:
                    p = hf_hub_download(repo, path, repo_type="dataset")
                    print(f"staged {tag} {path}", flush=True); return p
                except Exception as e:
                    print(f"stage {tag} {path} retry {attempt}: {str(e)[:110]}", flush=True); time.sleep(10)
            return None
        if stream_smol:
            for i in range(9):
                _stage("HuggingFaceTB/smoltalk", f"data/all/train-{i:05d}-of-00009.parquet", "smoltalk")
            cmd += ["--stream_smol", str(stream_smol)]
        if stream_mobile:
            _stage("google/mobile-actions", "dataset.jsonl", "mobile")
            cmd += ["--stream_mobile", str(stream_mobile)]
        if with_aria:
            jp = _stage("Aria-UI/Aria-UI_Data", "desktop/aria_ui_desktop_with_instructions.json", "aria-json")
            zp = _stage("Aria-UI/Aria-UI_Data", "desktop/screenshots.zip", "aria-zip")
            aria_jsonl = "/data/aria/desktop_grounding.jsonl"
            os.makedirs("/data/aria", exist_ok=True)
            if jp and not os.path.exists(aria_jsonl):
                from stream_data import aria_preprocess
                nrec = aria_preprocess(jp, aria_jsonl, max_samples=aria_max)
                print(f"aria_preprocess -> {aria_jsonl} ({nrec} grounding samples)", flush=True)
                runs_vol.commit()
            if jp and zp:
                cmd += ["--aria_jsonl", aria_jsonl, "--aria_zip", zp]
    if vision_id:
        # pre-cache the (new, high-res) SigLIP2 encoder ONCE here so 8 ranks don't race-download it at startup.
        import time
        from huggingface_hub import snapshot_download
        for attempt in range(6):
            try:
                snapshot_download(vision_id); print(f"pre-cached vision {vision_id}", flush=True); break
            except Exception as e:
                print(f"vision predownload retry {attempt}: {str(e)[:120]}", flush=True); time.sleep(8)
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def unified(stage2_run: str = "500M_vlm_stage2", audio_run: str = "500M_vlm_audio_stage1",
            n: int = 4, max_new: int = 40, joint_run: str = ""):
    """ONE model, three modalities: load stage-2 LLM + mm_projector (image & video) + audio_projector,
    then describe an image, a video (frames of it), and an audio clip."""
    import os, sys, io, json, zipfile, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from video import SiglipVideo
    from audio import ClapAudio
    from multimodal import MoEVLM, IMAGE_TOKEN, VIDEO_TOKEN, AUDIO_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from vlm_audio_data import ClothoAudio

    dev = torch.device("cuda")
    vis = SiglipVision(device=dev); vid = SiglipVideo(vis); aud = ClapAudio(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=vis.hidden, audio_dim=aud.hidden).to(dev)
    if joint_run:                                    # one checkpoint with all three (jointly trained)
        jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(jk["model"]); vlm.mm_projector.load_state_dict(jk["projector"])
        vlm.audio_projector.load_state_dict(jk["audio_projector"])
        print(f"UNIFIED (joint): all three from {joint_run}", flush=True)
    else:
        s2 = torch.load(f"/data/runs/{stage2_run}/model.pt", map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(s2["model"]); vlm.mm_projector.load_state_dict(s2["projector"])
        ap = torch.load(f"/data/runs/{audio_run}/audio_projector.pt", map_location=dev, weights_only=False)
        vlm.audio_projector.load_state_dict(ap["audio_projector"])
        print(f"UNIFIED: LLM+mm from {stage2_run}, audio from {audio_run}", flush=True)
    vlm.eval()
    tok = tiktoken.get_encoding("gpt2")

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen(sentinel, **feat):
        pre = tok.encode_ordinary("USER: Describe this in detail.\nASSISTANT:")
        ids = torch.tensor([[sentinel] + pre], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, **feat)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs).strip()

    laion = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    imgzip = zipfile.ZipFile("/llava/images.zip")
    clotho = ClothoAudio()
    for k in range(n):
        ex = laion[(k * 7919) % len(laion)]
        img = Image.open(io.BytesIO(imgzip.read(ex["image"]))).convert("RGB")
        ifeats = vis.encode([img])
        vfeats = vid.encode_frames([img] * 2)            # 2 frames x 729 = 1458 raw tokens (fits 2048 ctx)
        wav, agt = clotho.raw((k * 619) % len(clotho))
        afeats = aud.encode([wav])
        print(f"\n=== sample {k} ===", flush=True)
        print(f"[IMAGE]  PRED: {gen(IMAGE_TOKEN, image_features=ifeats)}", flush=True)
        print(f"[VIDEO]  PRED: {gen(VIDEO_TOKEN, video_features=vfeats)}", flush=True)
        print(f"[AUDIO GT: {str(agt)[:55]}]\n[AUDIO]  PRED: {gen(AUDIO_TOKEN, audio_features=afeats)}", flush=True)
    print("\nUNIFIED DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=40 * 60, secrets=[HF])
def unified5(joint_run: str = "500M_vlm_joint5", n: int = 4, max_new: int = 48):
    """ONE checkpoint, FIVE paths: image / video / audio(sound) / speech(spoken-QA) / text. Loads the
    5-path joint model and exercises each modality on held-out data."""
    import os, sys, io, json, zipfile, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from video import SiglipVideo
    from audio import ClapAudio
    from speech import WhisperSpeech
    from multimodal import MoEVLM, IMAGE_TOKEN, VIDEO_TOKEN, AUDIO_TOKEN, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from vlm_audio_data import ClothoAudio
    from vlm_va_data import VoiceAssistantQA

    dev = torch.device("cuda")
    vis = SiglipVision(device=dev); vid = SiglipVideo(vis); aud = ClapAudio(device=dev)
    spk = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=vis.hidden, audio_dim=aud.hidden, speech_dim=spk.hidden).to(dev)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(jk["model"]); vlm.mm_projector.load_state_dict(jk["projector"])
    vlm.audio_projector.load_state_dict(jk["audio_projector"])
    vlm.speech_projector.load_state_dict(jk["speech_projector"])
    vlm.eval()
    print(f"UNIFIED-5 (joint): image/video/audio/speech/text from {joint_run}", flush=True)
    tok = tiktoken.get_encoding("gpt2")

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen(sentinel, prompt="USER: Describe this in detail.\nASSISTANT:", **feat):
        pre = tok.encode_ordinary(prompt) if prompt else []
        ids = torch.tensor([[sentinel] + pre], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, **feat)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs).strip()

    laion = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    imgzip = zipfile.ZipFile("/llava/images.zip")
    clotho = ClothoAudio()
    va = VoiceAssistantQA(max_shards=1)
    # text-only sanity: no sentinel, plain prompt
    print(f"\n[TEXT] Q: 'The capital of France is' -> {gen(EOT, prompt='The capital of France is')}", flush=True)
    for k in range(n):
        ex = laion[(k * 7919) % len(laion)]
        img = Image.open(io.BytesIO(imgzip.read(ex["image"]))).convert("RGB")
        ifeats = vis.encode([img]); vfeats = vid.encode_frames([img] * 2)
        wav, agt = clotho.raw((k * 619) % len(clotho)); afeats = aud.encode([wav])
        swav, sans, sq = va.raw((k * 877) % len(va)); sfeats = spk.encode([swav])
        print(f"\n=== sample {k} ===", flush=True)
        print(f"[IMAGE]  {gen(IMAGE_TOKEN, image_features=ifeats)}", flush=True)
        print(f"[VIDEO]  {gen(VIDEO_TOKEN, video_features=vfeats)}", flush=True)
        print(f"[AUDIO GT {str(agt)[:45]}] {gen(AUDIO_TOKEN, audio_features=afeats)}", flush=True)
        print(f"[SPEECH Q {sq[:45]}]\n  GT: {sans[:90]}\n  PRED: {gen(SPEECH_TOKEN, prompt='', speech_features=sfeats)}", flush=True)
    print("\nUNIFIED-5 DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def image_voice(joint_run: str = "500M_vlm_joint5", n: int = 4, max_new: int = 48):
    """ZERO-SHOT image + VOICE question: TTS a spoken question, feed [IMAGE][SPEECH] to the 5-path model,
    and see whether the answer is grounded in the image (the model was never trained on image+speech together)."""
    import os, sys, io, json, zipfile, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from speech import WhisperSpeech, WHISPER_SR
    from multimodal import MoEVLM, IMAGE_TOKEN, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    vis = SiglipVision(device=dev); spk = WhisperSpeech(device=dev)
    # TTS to synthesize the spoken question (facebook/mms-tts-eng = VITS, 16 kHz — Whisper-native)
    from transformers import VitsModel, AutoTokenizer
    tts = VitsModel.from_pretrained("facebook/mms-tts-eng").to(dev).eval()
    ttok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    tts_sr = tts.config.sampling_rate

    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=vis.hidden, speech_dim=spk.hidden).to(dev)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(jk["model"]); vlm.mm_projector.load_state_dict(jk["projector"])
    vlm.speech_projector.load_state_dict(jk["speech_projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    print(f"IMAGE+VOICE (zero-shot) from {joint_run} | tts_sr={tts_sr}", flush=True)

    @torch.no_grad()
    def synth(text):
        inp = ttok(text, return_tensors="pt").to(dev)
        wav = tts(**inp).waveform[0].float().cpu().numpy()
        if tts_sr != WHISPER_SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=tts_sr, target_sr=WHISPER_SR)
        return wav

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen(ifeats, sfeats, tail=""):
        # image (context) then spoken question, then optional text cue -> answer
        pre = tok.encode_ordinary(tail) if tail else []
        ids = torch.tensor([[IMAGE_TOKEN, SPEECH_TOKEN] + pre], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, image_features=ifeats, speech_features=sfeats)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs).strip()

    questions = ["What is in this image?", "What is the main object in the picture?",
                 "How many people are in the image?", "What colors do you see?"]
    laion = json.load(open("/llava/blip_laion_cc_sbu_558k.json"))
    imgzip = zipfile.ZipFile("/llava/images.zip")
    for k in range(n):
        ex = laion[(k * 7919) % len(laion)]
        img = Image.open(io.BytesIO(imgzip.read(ex["image"]))).convert("RGB")
        q = questions[k % len(questions)]
        ifeats = vis.encode([img]); sfeats = spk.encode([synth(q)])
        gt = ex["conversations"][1]["value"].strip().replace("\n", " ")[:70]
        print(f"\n=== sample {k} | IMAGE: {gt} ===", flush=True)
        print(f"  SPOKEN-Q: {q}", flush=True)
        print(f"  bare [IMAGE][SPEECH]      -> {gen(ifeats, sfeats)}", flush=True)
        print(f"  + ' ASSISTANT:' text cue  -> {gen(ifeats, sfeats, tail=' ASSISTANT:')}", flush=True)
    print("\nIMAGE+VOICE DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def speech_smoke():
    """End-to-end speech path on synthetic 16kHz audio: Whisper -> project -> splice <speech> -> 500M -> loss/grad."""
    import os, sys, numpy as np, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from speech import WhisperSpeech, WHISPER_SR
    from multimodal import MoEVLM, SPEECH_TOKEN
    from generate import load_model

    dev = torch.device("cuda")
    torch.manual_seed(0)
    enc = WhisperSpeech(device=dev)
    wav = (np.random.randn(WHISPER_SR * 6).astype("float32")) * 0.1     # 6s synthetic mono @ 16k
    feats = enc.encode([wav])                                          # (1, 1500/stack, d*stack)
    print(f"Whisper hidden={enc.hidden} (d{enc.d} x stack{enc.stack})  features={tuple(feats.shape)}", flush=True)

    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    print(f"backbone {BACKBONE}: d{cfg.d_model} val={vloss}", flush=True)
    llm.train()
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    tok = tiktoken.get_encoding("gpt2")
    words = tok.encode_ordinary("the quick brown fox jumps over the lazy dog")
    logical = [SPEECH_TOKEN] + words + [50256]
    ids = torch.tensor([logical[:-1]], device=dev)
    tgt = torch.tensor([logical[1:]], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = vlm(ids, speech_features=feats, targets=tgt)
    loss.backward()
    g = vlm.speech_projector.net[0].weight.grad
    print(f"merged len={feats.shape[1] + len(words)}  loss={loss.item():.4f}  "
          f"speech-projector grad finite={bool(torch.isfinite(g).all())}", flush=True)
    assert torch.isfinite(loss) and torch.isfinite(g).all()
    print("SPEECH SMOKE OK", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def train_speech_stage1(max_steps: int = 1500, micro: int = 12, lr: float = 1e-3, warmup: int = 100,
                        save_name: str = "500M_vlm_speech_stage1", repo: str = "openslr/librispeech_asr",
                        max_shards: int = 6, log_every: int = 20):
    """Speech stage-1 alignment: train ONLY the speech projector on LibriSpeech ASR; Whisper + LLM frozen.
    Teaches the projector to map Whisper speech features -> spoken-word tokens (read-out transcription)."""
    import os, sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from torch.utils.data import DataLoader
    from speech import WhisperSpeech
    from multimodal import MoEVLM
    from generate import load_model
    from vlm_speech_data import LibriSpeechASR, speech_collate

    dev = torch.device("cuda")
    torch.manual_seed(0); torch.set_float32_matmul_precision("high")
    enc = WhisperSpeech(device=dev)
    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    vlm.set_llm_trainable(False); llm.set_bias_update_rate(0.0)
    n_proj = sum(p.numel() for p in vlm.speech_projector.parameters())
    print(f"backbone d{cfg.d_model} val={vloss:.3f} | Whisper hidden={enc.hidden} | "
          f"speech-projector={n_proj/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(vlm.speech_projector.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    ds = LibriSpeechASR(repo, max_shards=max_shards)
    dl = DataLoader(ds, batch_size=micro, shuffle=True, num_workers=4, collate_fn=speech_collate,
                    drop_last=True, persistent_workers=True)
    vlm.train()
    step, t0, run, last, done = 0, time.time(), 0.0, float("nan"), False
    while not done:
        for wavs, ids, tgt in dl:
            ids, tgt = ids.to(dev), tgt.to(dev)
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = enc.encode(wavs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = vlm(ids, speech_features=feats, targets=tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.speech_projector.parameters(), 1.0)
            opt.step()
            last = loss.item(); run += last
            if step % log_every == 0:
                print(f"step {step:5d} | loss {last:.4f} | avg {run/(step+1):.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e} | {(time.time()-t0)/(step+1)*1000:.0f}ms/step", flush=True)
            step += 1
            if step >= max_steps:
                done = True; break

    out = f"/data/runs/{save_name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"speech_projector": vlm.speech_projector.state_dict(), "speech_dim": enc.hidden,
                "whisper_stack": enc.stack, "backbone": BACKBONE, "steps": step}, f"{out}/speech_projector.pt")
    runs_vol.commit()
    print(f"saved speech projector -> {out}/speech_projector.pt (final {last:.4f})", flush=True)
    return {"final_loss": last, "steps": step}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def speech_parity(joint_run: str = "500M_vlm_joint12",
                  audio_url: str = "https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav",
                  prompt: str = "What is being said in the audio?", max_new: int = 80):
    """PARITY: run the SAME jfk.wav through the PyTorch joint12 speech path (built on the ctx8k theta1e6
    backbone, plain greedy = mtmd --temp 0) and print the response. If PyTorch also answers vaguely/wrongly,
    the GGUF audio port is FAITHFUL (the limit is joint12's weak speech, per notes) — not a stack/feature bug."""
    import os, sys, urllib.request, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken, librosa
    from speech import WhisperSpeech, WHISPER_SR
    from multimodal import MoEVLM, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    enc = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model("/data/runs/500M_ctx8k/model.pt", dev)     # theta1e6 8k cfg (NOT ctx2048!)
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(jk["model"]); vlm.speech_projector.load_state_dict(jk["speech_projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    urllib.request.urlretrieve(audio_url, "/tmp/p.wav")
    wav, _ = librosa.load("/tmp/p.wav", sr=WHISPER_SR)

    @torch.no_grad()
    def gen():
        feats = enc.encode([wav])
        ids = torch.tensor([[SPEECH_TOKEN] + tok.encode_ordinary(f"USER: {prompt}\nASSISTANT:")], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, speech_features=feats)
            outs = []
            for _ in range(max_new):                                       # plain greedy, no rep-pen (= --temp 0)
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs).strip()

    print(f"\n[PARITY {joint_run}] prompt 'USER: {prompt}\\nASSISTANT:'", flush=True)
    print(f"  PyTorch PRED: {gen()}", flush=True)
    print("  GGUF    PRED: (compare to mtmd_audio output)", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def caption_speech(speech_run: str = "500M_vlm_speech_stage1", n: int = 8, max_new: int = 48,
                   repo: str = "openslr/librispeech_asr", split_match: str = "test,clean"):
    """Transcribe a few held-out LibriSpeech clips with the speech VLM; print predicted vs ground-truth."""
    import os, sys, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from speech import WhisperSpeech
    from multimodal import MoEVLM, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from vlm_speech_data import LibriSpeechASR

    dev = torch.device("cuda")
    enc = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    ck = torch.load(f"/data/runs/{speech_run}/speech_projector.pt", map_location=dev, weights_only=False)
    vlm.speech_projector.load_state_dict(ck["speech_projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    ds = LibriSpeechASR(repo, match=tuple(split_match.split(",")), max_shards=1)

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen(wav):
        feats = enc.encode([wav])
        ids = torch.tensor([[SPEECH_TOKEN]], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, speech_features=feats)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs)

    for k in range(n):
        i = (k * 311) % len(ds)
        wav, gt = ds.raw(i)
        print(f"\n[{i}] GT:   {gt[:90]}\n     PRED: {gen(wav).strip()}", flush=True)
    print("\nSPEECH TRANSCRIBE DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def train_speech_sft(max_steps: int = 1200, micro: int = 8, lr: float = 2e-5, proj_lr: float = 1e-4,
                     warmup: int = 80, speech_run: str = "500M_vlm_speech_stage1",
                     save_name: str = "500M_vlm_speech_sft", repo: str = "gpt-omni/VoiceAssistant-400K",
                     max_shards: int = 4, log_every: int = 20):
    """Spoken-QA SFT: co-train speech_projector + LLM on VoiceAssistant-400K so the model ANSWERS a spoken
    question (not transcribe it). Init speech_projector from the ASR stage-1. Whisper frozen. Single H100."""
    import os, sys, math, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from torch.utils.data import DataLoader
    from speech import WhisperSpeech
    from multimodal import MoEVLM
    from generate import load_model
    from vlm_va_data import VoiceAssistantQA, va_collate

    dev = torch.device("cuda")
    torch.manual_seed(0); torch.set_float32_matmul_precision("high")
    enc = WhisperSpeech(device=dev)
    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    sp = torch.load(f"/data/runs/{speech_run}/speech_projector.pt", map_location=dev, weights_only=False)
    vlm.speech_projector.load_state_dict(sp["speech_projector"])
    vlm.set_llm_trainable(True); llm.set_bias_update_rate(0.0)
    proj_ids = {id(p) for p in vlm.speech_projector.parameters()}
    proj = [p for p in vlm.parameters() if p.requires_grad and id(p) in proj_ids]
    llmp = [p for p in vlm.parameters() if p.requires_grad and id(p) not in proj_ids]
    opt = torch.optim.AdamW([{"params": llmp, "lr": lr}, {"params": proj, "lr": proj_lr}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    base = [lr, proj_lr]
    print(f"speech-SFT init: speech_projector from {speech_run} | LLM trainable | "
          f"backbone val={vloss:.3f}", flush=True)

    def lr_at(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, max_steps - warmup)))

    ds = VoiceAssistantQA(repo, max_shards=max_shards)
    dl = DataLoader(ds, batch_size=micro, shuffle=True, num_workers=4, collate_fn=va_collate,
                    drop_last=True, persistent_workers=True)
    vlm.train()
    step, t0, run, last, done = 0, time.time(), 0.0, float("nan"), False
    while not done:
        for wavs, ids, tgt in dl:
            ids, tgt = ids.to(dev), tgt.to(dev)
            m = lr_at(step)
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * m
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feats = enc.encode(wavs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = vlm(ids, speech_features=feats, targets=tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in vlm.parameters() if p.requires_grad], 1.0)
            opt.step()
            last = loss.item(); run += last
            if step % log_every == 0:
                print(f"step {step:5d} | loss {last:.4f} | avg {run/(step+1):.4f} | "
                      f"lr {opt.param_groups[0]['lr']:.2e} | {(time.time()-t0)/(step+1)*1000:.0f}ms/step", flush=True)
            step += 1
            if step >= max_steps:
                done = True; break

    out = f"/data/runs/{save_name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"model": vlm.llm.state_dict(), "speech_projector": vlm.speech_projector.state_dict(),
                "speech_dim": enc.hidden, "whisper_stack": enc.stack,
                "config": {**cfg.to_dict(), "preset": "500M"}, "backbone": BACKBONE}, f"{out}/model.pt")
    runs_vol.commit()
    print(f"saved speech-SFT -> {out}/model.pt (final {last:.4f})", flush=True)
    return {"final_loss": last, "steps": step}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def ask_speech(sft_run: str = "500M_vlm_speech_sft", n: int = 6, max_new: int = 64,
               repo: str = "gpt-omni/VoiceAssistant-400K"):
    """Ask the spoken-QA model held-out spoken questions; print the (text of the) question, GT and PRED answer."""
    import os, sys, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from speech import WhisperSpeech
    from multimodal import MoEVLM, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from vlm_va_data import VoiceAssistantQA

    dev = torch.device("cuda")
    enc = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    ck = torch.load(f"/data/runs/{sft_run}/model.pt", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(ck["model"]); vlm.speech_projector.load_state_dict(ck["speech_projector"])
    vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    ds = VoiceAssistantQA(repo, max_shards=1)

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen(wav):
        feats = enc.encode([wav])
        ids = torch.tensor([[SPEECH_TOKEN]], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, speech_features=feats)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                for b in _banned(outs):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs)

    for k in range(n):
        i = (k * 877) % len(ds)
        wav, gt, q = ds.raw(i)
        print(f"\n[{i}] SPOKEN-Q: {q[:80]}\n   GT:   {gt[:110]}\n   PRED: {gen(wav).strip()[:160]}", flush=True)
    print("\nSPOKEN-QA DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=4 * 60 * 60, secrets=[HF])
def speech_wer(speech_run: str = "500M_vlm_speech_stage1", limit: int = 0, max_new: int = 100,
               repo: str = "openslr/librispeech_asr", split_match: str = "test,clean",
               joint_run: str = "", backbone: str = ""):
    """Word Error Rate on LibriSpeech test-clean: plain greedy decode (no rep-penalty, the honest ASR
    setting), word-level Levenshtein vs ground truth, normalized (lowercase, strip punctuation)."""
    import os, sys, re, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from speech import WhisperSpeech
    from multimodal import MoEVLM, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from vlm_speech_data import LibriSpeechASR

    dev = torch.device("cuda")
    enc = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt" if backbone else BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, speech_dim=enc.hidden).to(dev)
    if joint_run:                                          # full joint ckpt: its LLM + speech_projector
        jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(jk["model"]); vlm.speech_projector.load_state_dict(jk["speech_projector"])
    else:
        ck = torch.load(f"/data/runs/{speech_run}/speech_projector.pt", map_location=dev, weights_only=False)
        vlm.speech_projector.load_state_dict(ck["speech_projector"])
    vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    ds = LibriSpeechASR(repo, match=tuple(split_match.split(",")), max_shards=0)

    def norm(s):
        return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()

    def edit(r, h):                                   # word-level Levenshtein
        d = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(h) + 1):
                cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
                prev = cur
        return d[len(h)]

    @torch.no_grad()
    def transcribe(wav):
        feats = enc.encode([wav])
        ids = torch.tensor([[SPEECH_TOKEN]], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, speech_features=feats)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs)

    N = len(ds) if not limit else min(limit, len(ds))
    tot_err, tot_words, t0 = 0, 0, time.time()
    for i in range(N):
        wav, gt = ds.raw(i)
        r, h = norm(gt), norm(transcribe(wav))
        tot_err += edit(r, h); tot_words += len(r)
        if i < 4 or (i + 1) % 200 == 0:
            print(f"[{i+1}/{N}] WER so far {tot_err/max(1,tot_words)*100:.2f}% | "
                  f"{(time.time()-t0)/(i+1)*1000:.0f}ms/clip\n   GT:   {gt[:80]}\n   HYP:  {' '.join(h)[:80]}",
                  flush=True)
    wer = tot_err / max(1, tot_words) * 100
    print(f"\n=== WER = {wer:.2f}%  over {N} clips, {tot_words} words ({speech_run}) ===", flush=True)
    return {"wer": wer, "clips": N, "words": tot_words}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=3 * 60 * 60, secrets=[HF])
def hotpotqa_eval(joint_run: str = "500M_vlm_joint12", backbone: str = "500M_ctx8k", limit: int = 500,
                  max_new: int = 64):
    """HotPotQA (distractor) reading EM/F1: feed the 10 context paragraphs + the multi-hop question (fits in
    8k ctx), greedy-decode a short answer, score SQuAD-style normalized exact-match + token-F1. A showcase for
    the 8k context (closed reading-comprehension, like the Baguettotron-table setting)."""
    import os, sys, re, string, collections, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from datasets import load_dataset
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt", dev)
    ck = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
    llm.load_state_dict(ck["model"]); llm.eval()
    tok = tiktoken.get_encoding("gpt2")

    def _norm(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(c for c in s if c not in string.punctuation)
        return " ".join(s.split())

    def _f1(pred, gold):
        p, g = _norm(pred).split(), _norm(gold).split()
        if not p or not g:
            return float(p == g)
        common = collections.Counter(p) & collections.Counter(g)
        ns = sum(common.values())
        if ns == 0:
            return 0.0
        prec, rec = ns / len(p), ns / len(g)
        return 2 * prec * rec / (prec + rec)

    @torch.no_grad()
    def gen(prompt_ids):
        ids = torch.tensor([prompt_ids], device=dev)
        out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(max_new):
                lg = llm(ids)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                out.append(t)
                ids = torch.cat([ids, torch.tensor([[t]], device=dev)], dim=1)
        return tok.decode(out)

    import re as _re
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    N = min(limit or len(ds), len(ds))
    em = f1s = contains = 0.0                              # EM(strict) + token-F1 + CONTAINMENT (gold in full resp)
    for i in range(N):
        ex = ds[i]
        ctx = ""
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            ctx += f"{title}: {''.join(sents)}\n"
        # joint12's NATIVE chat format (it was trained USER:/ASSISTANT:, not Question:/Answer:)
        prompt = f"USER: Use the context to answer with a short answer only.\n{ctx}\nQuestion: {ex['question']}\nASSISTANT:"
        ids = tok.encode_ordinary(prompt)[-7000:]  # 8k ctx holds it
        full = gen(ids).strip()
        pred = full.split("\n")[0].strip()
        gold = ex["answer"]
        em += float(_norm(pred) == _norm(gold))
        f1s += max(_f1(pred, gold), _f1(full, gold))      # best of first-line / full (verbose-tolerant)
        ng = _norm(gold)
        contains += bool(ng and _re.search(rf"\b{_re.escape(ng)}\b", _norm(full)))   # lenient: answer present?
        if i < 4:
            print(f"  Q: {ex['question'][:70]} | GOLD: {gold} | PRED: {full[:90]}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N}] EM {em/(i+1)*100:.1f} F1 {f1s/(i+1)*100:.1f} Contains {contains/(i+1)*100:.1f}",
                  flush=True)
    print(f"\n=== HotPotQA (distractor, n={N}): EM {em/N*100:.2f}  F1 {f1s/N*100:.2f}  "
          f"Contains {contains/N*100:.2f} ===", flush=True)
    return {"em": em / N * 100, "f1": f1s / N * 100, "contains": contains / N * 100, "n": N}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=4 * 60 * 60, secrets=[HF])
def vqa_lenient(joint_run: str = "500M_vlm_joint12", backbone: str = "500M_ctx8k",
                vision_id: str = "google/siglip2-so400m-patch16-512",
                tasks: str = "pope,vqav2,gqa", limit: int = 1000, max_new: int = 32):
    """Containment-aware re-score of POPE/VQAv2/GQA: generate the FULL answer (joint12 is verbose from chat
    training) and score leniently — gold answer appearing in the response (POPE: leading yes/no). Fixes the
    single-word-extraction artifact in the lmms-eval harness that mis-scored joint12's verbose answers."""
    import os, sys, re, string, io, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from datasets import load_dataset
    from PIL import Image
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    enc = SiglipVision(model_id=vision_id, device=dev) if vision_id else SiglipVision(device=dev)
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt", dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    ck = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(ck["model"]); vlm.mm_projector.load_state_dict(ck["projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")

    def norm(s):
        s = s.lower().strip()
        s = "".join(c for c in s if c not in string.punctuation)
        return " ".join(s.split())

    @torch.no_grad()
    def answer(image, q):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = enc.encode([image.convert("RGB")])
            pre = tok.encode_ordinary(f"USER: {q}\nASSISTANT:")
            cur, _ = vlm.build_inputs_embeds(torch.tensor([[IMAGE_TOKEN] + pre], device=dev), image_features=feats)
            out = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                out.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(out).strip()

    res = {}
    for task in [t for t in tasks.split(",") if t]:
        if task == "pope":
            ds = load_dataset("lmms-lab/POPE", split="test")
            N = min(limit or len(ds), len(ds)); tp = fp = tn = fn = 0
            for i in range(N):
                ex = ds[i]; r = norm(answer(ex["image"], ex["question"]))
                pred = "yes" if r.startswith("yes") else "no"
                gold = norm(ex["answer"])
                tp += (pred == "yes" and gold == "yes"); fp += (pred == "yes" and gold == "no")
                tn += (pred == "no" and gold == "no"); fn += (pred == "no" and gold == "yes")
            acc = (tp + tn) / max(1, N); prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            res["pope"] = {"acc": acc * 100, "f1": f1 * 100, "n": N}
        elif task == "vqav2":
            ds = load_dataset("lmms-lab/VQAv2", split="validation")
            N = min(limit or len(ds), len(ds)); sc = 0.0
            for i in range(N):
                ex = ds[i]; r = norm(answer(ex["image"], ex["question"]))
                golds = [norm(a["answer"]) if isinstance(a, dict) else norm(a) for a in ex["answers"]]
                m = sum(1 for g in golds if g and re.search(rf"\b{re.escape(g)}\b", r))
                sc += min(m / 3.0, 1.0)
            res["vqav2"] = {"vqa_acc": sc / N * 100, "n": N}
        elif task == "gqa":
            try:
                ins = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
                imgs = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
                id2img = {row["id"]: row["image"] for row in imgs}
                N = min(limit or len(ins), len(ins)); hit = 0
                for i in range(N):
                    ex = ins[i]; im = id2img.get(ex["imageId"])
                    if im is None:
                        continue
                    r = norm(answer(im, ex["question"])); g = norm(ex["answer"])
                    hit += bool(g and re.search(rf"\b{re.escape(g)}\b", r))
                res["gqa"] = {"acc": hit / N * 100, "n": N}
            except Exception as e:
                res["gqa"] = {"error": str(e)[:120]}
        print(f"  {task}: {res[task]}", flush=True)
    print(f"\n=== VQA LENIENT (containment) {joint_run} ===\n{res}", flush=True)
    return res


@app.function(image=lmms_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=6 * 60 * 60, secrets=[HF])
def vlm_eval_run(stage2_run: str = "500M_vlm_stage2", joint_run: str = "", tasks: str = "pope",
                 limit: int = 0, max_new: int = 64, backbone: str = "", vision_id: str = ""):
    """Run the lmms-eval harness (POPE/GQA/VQAv2) on our MoE-VLM via the registered `moe_vlm` model.
    Mirrors modal_train.lm_eval_run but for the multimodal harness."""
    import os, sys, json
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import torch
    from lmms_eval.evaluator import simple_evaluate
    from lmms_eval.models import AVAILABLE_MODELS
    # lmms-eval 0.3.0 resolves model names via AVAILABLE_MODELS (a dotted module.Class path), NOT the
    # @register_model registry. /root/moe-lab is on sys.path, so point it straight at our wrapper.
    AVAILABLE_MODELS["moe_vlm"] = "vlm_eval_harness.MoEVLMHarness"

    torch.set_float32_matmul_precision("high")
    margs = f"stage2_run={stage2_run},max_new={max_new}"
    if joint_run:
        margs += f",joint_run={joint_run}"
    if backbone:
        margs += f",backbone=/data/runs/{backbone}/model.pt"
    if vision_id:
        margs += f",vision_id={vision_id}"
    task_list = [t for t in tasks.split(",") if t]
    print(f"[vlm_eval] model_args={margs} | tasks={task_list} | limit={limit or 'full'}", flush=True)
    res = simple_evaluate(model="moe_vlm", model_args=margs, tasks=task_list,
                          limit=(limit or None), batch_size=1)

    print("\n=== lmms-eval RESULTS ===", flush=True)
    summary = {}
    for task, metrics in res["results"].items():
        nice = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not k.endswith("_stderr")}
        summary[task] = nice
        print(f"  {task}: " + "  ".join(f"{k}={v:.4f}" for k, v in nice.items()), flush=True)

    run = joint_run or stage2_run
    tag = "_".join(task_list)                              # task-aware filename so parallel runs don't clobber
    try:
        with open(f"/data/runs/{run}/vlm_eval_{tag}.json", "w") as f:
            json.dump({"run": run, "tasks": task_list, "limit": limit, "results": res["results"]},
                      f, indent=2, default=str)
        runs_vol.commit()
        print(f"saved -> /data/runs/{run}/vlm_eval_{tag}.json", flush=True)
    except Exception as e:
        print(f"(could not save vlm_eval.json: {e})", flush=True)
    return summary


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=4 * 60 * 60, secrets=[HF])
def prep_speech_tool(n: int = 14000, val: int = 1000, repo: str = "nvidia/Nemotron-Agentic-v1", max_q: int = 180):
    """TTS the spoken-query tool loops: Nemotron query->call->result->answer, with the query synthesized to
    audio (facebook/mms-tts-eng, 16 kHz). Writes /data/tools/speech_tool_{train,val}.jsonl (wav b64 + segs)."""
    import os, sys, json, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from huggingface_hub import hf_hub_download
    from transformers import VitsModel, AutoTokenizer
    from speech_tool_data import extract_speech_tool, wav_to_b64

    dev = torch.device("cuda")
    tts = VitsModel.from_pretrained("facebook/mms-tts-eng").to(dev).eval()
    ttok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    sr = tts.config.sampling_rate
    path = hf_hub_download(repo, "data/tool_calling.jsonl", repo_type="dataset", local_dir="/cache/nemotron")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            try:
                ex = extract_speech_tool(json.loads(line))
            except Exception:
                ex = None
            if not ex:
                continue
            query, tools_json, post = ex
            if not (3 < len(query) <= max_q):
                continue
            try:
                with torch.no_grad():
                    wav = tts(**ttok(query, return_tensors="pt").to(dev)).waveform[0].float().cpu().numpy()
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            except Exception:
                continue
            rows.append({"wav_b64": wav_to_b64(wav), "tools": tools_json, "post": post})
            if len(rows) % 1000 == 0:
                print(f"  tts {len(rows)}/{n}", flush=True)
    import random
    random.Random(0).shuffle(rows)
    vr, tr = rows[:val], rows[val:]
    os.makedirs("/data/tools", exist_ok=True)
    for name, rs in [("speech_tool_train.jsonl", tr), ("speech_tool_val.jsonl", vr)]:
        with open(f"/data/tools/{name}", "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    runs_vol.commit()
    print(f"speech-tool: train={len(tr)} val={len(vr)} -> /data/tools", flush=True)
    return {"train": len(tr), "val": len(vr)}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def agent_demo(joint_run: str = "500M_vlm_joint8", query: str = "What's the weather in Tokyo right now?",
               ckpt: str = "model.pt", backbone: str = ""):
    """The omni-agent with the forced-call policy baked in (agent.py): text AND spoken query -> tool call
    -> result -> summary. Uses a dummy tool result for the demo."""
    import os, sys, json, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from speech import WhisperSpeech, WHISPER_SR
    from multimodal import MoEVLM
    from generate import load_model
    from agent import agent_tool_call, agent_summarize

    dev = torch.device("cuda")
    spk = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt" if backbone else BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, audio_dim=768, speech_dim=spk.hidden).to(dev)
    jk = torch.load(f"/data/runs/{joint_run}/{ckpt}", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(jk["model"]); vlm.mm_projector.load_state_dict(jk["projector"])
    vlm.audio_projector.load_state_dict(jk["audio_projector"])
    vlm.speech_projector.load_state_dict(jk["speech_projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")

    tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
              "parameters": {"type": "object", "properties": {
                  "city": {"type": "string", "description": "City name"},
                  "unit": {"type": "string", "description": "celsius or fahrenheit"}}, "required": ["city"]}}]
    tools_json = json.dumps(tools, separators=(",", ":"))
    result = {"temperature": 14, "unit": "celsius", "condition": "light rain", "humidity": "82%"}
    print(f"loaded {joint_run} | tools=[get_weather]\n", flush=True)

    # ---- TEXT (forced-call policy) ----
    tcall = agent_tool_call(vlm, tok, dev, tools_json, query=query)
    tans = agent_summarize(vlm, tok, dev, tools_json, tcall, result, query=query)
    print("=== TEXT AGENT ===", flush=True)
    print(f"  QUERY:  {query}\n  CALL:   {tcall}\n  RESULT: {json.dumps(result)}\n  ANSWER: {tans}\n", flush=True)

    # ---- SPEECH (same query, spoken) ----
    from transformers import VitsModel, AutoTokenizer
    tts = VitsModel.from_pretrained("facebook/mms-tts-eng").to(dev).eval()
    ttok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    with torch.no_grad():
        wav = tts(**ttok(query, return_tensors="pt").to(dev)).waveform[0].float().cpu().numpy()
    if tts.config.sampling_rate != WHISPER_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=tts.config.sampling_rate, target_sr=WHISPER_SR)
    sf = spk.encode([wav])
    scall = agent_tool_call(vlm, tok, dev, tools_json, speech_features=sf)
    sans = agent_summarize(vlm, tok, dev, tools_json, scall, result, speech_features=sf)
    print("=== SPEECH AGENT ===", flush=True)
    print(f"  SPOKEN: {query}\n  CALL:   {scall}\n  RESULT: {json.dumps(result)}\n  ANSWER: {sans}", flush=True)
    print("\nAGENT DEMO DONE", flush=True)


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def explain_image(joint_run: str = "500M_vlm_joint9", img: str = "", prompt: str = "", max_new: int = 96,
                  ckpt: str = "model.pt", backbone: str = "", vision_id: str = ""):
    """Describe/explain ONE arbitrary image with the joint VLM. `img` is an http(s) URL (fetched in-container)
    or a path on the llava-data volume (e.g. an uploaded file at /llava/<name>). Uses the image-describe path
    (SigLIP2 -> mm_projector -> LLM) with rep-penalty + no-repeat-3gram decoding."""
    import os, sys, io, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from PIL import Image
    import tiktoken
    from vision import SiglipVision
    from multimodal import MoEVLM, IMAGE_TOKEN
    from generate import load_model, GPT2_VALID, EOT

    dev = torch.device("cuda")
    enc = SiglipVision(model_id=vision_id, device=dev) if vision_id else SiglipVision(device=dev)
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt" if backbone else BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=enc.hidden).to(dev)
    ck = torch.load(f"/data/runs/{joint_run}/{ckpt}", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(ck["model"]); vlm.mm_projector.load_state_dict(ck["projector"]); vlm.eval()
    tok = tiktoken.get_encoding("gpt2")

    if img.startswith("http"):
        import requests
        image = Image.open(io.BytesIO(requests.get(img, timeout=20).content)).convert("RGB")
    else:
        image = Image.open(img if img.startswith("/") else f"/llava/{img}").convert("RGB")
    q = prompt or "Describe this image in detail."
    pre = tok.encode_ordinary(f"USER: {q}\nASSISTANT:")
    print(f"loaded {joint_run} | img={img} | prompt: USER: {q}\n", flush=True)

    def _banned(prev, n=3):
        if len(prev) < n:
            return []
        seen = {}
        for j in range(len(prev) - n + 1):
            seen.setdefault(tuple(prev[j:j + n - 1]), []).append(prev[j + n - 1])
        return seen.get(tuple(prev[-(n - 1):]), [])

    @torch.no_grad()
    def gen(rep_pen=1.3):
        feats = enc.encode([image])
        ids = torch.tensor([[IMAGE_TOKEN] + pre], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(ids, image_features=feats)
            outs = []
            for _ in range(max_new):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / rep_pen, v * rep_pen)
                for b in _banned(outs, 3):
                    lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                e = vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)
                cur = torch.cat([cur, e], dim=1)
        return tok.decode(outs)

    ans = gen()
    print("=== EXPLANATION ===", flush=True)
    print(ans.strip(), flush=True)
    print("\nEXPLAIN DONE", flush=True)
    return {"img": img, "prompt": q, "answer": ans.strip()}


@app.function(image=vlm_image, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=30 * 60, secrets=[HF])
def demo_weather(joint_run: str = "500M_vlm_joint6", max_new: int = 64):
    """joint6 on a weather request, two ways: (1) TOOLS+text query -> tool call -> dummy result -> final
    answer; (2) the SAME question spoken (TTS -> Whisper -> speech path)."""
    import os, sys, json, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from speech import WhisperSpeech, WHISPER_SR
    from multimodal import MoEVLM, SPEECH_TOKEN
    from generate import load_model, GPT2_VALID, EOT
    from tool_decode import constrained_tool_gen, _free_greedy

    dev = torch.device("cuda")
    spk = WhisperSpeech(device=dev)
    llm, cfg, _, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=1152, audio_dim=768, speech_dim=spk.hidden).to(dev)
    jk = torch.load(f"/data/runs/{joint_run}/model.pt", map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(jk["model"]); vlm.mm_projector.load_state_dict(jk["projector"])
    vlm.audio_projector.load_state_dict(jk["audio_projector"])
    vlm.speech_projector.load_state_dict(jk["speech_projector"])
    vlm.eval()
    tok = tiktoken.get_encoding("gpt2")
    print(f"loaded {joint_run}", flush=True)

    # ---- (1) TEXT + TOOLS ----
    tools = [{"name": "get_current_temperature",
              "description": "Get the current temperature for a given city.",
              "parameters": {"type": "object", "properties": {
                  "city": {"type": "string", "description": "City name"},
                  "unit": {"type": "string", "description": "celsius or fahrenheit"}}, "required": ["city"]}}]
    tools_json = json.dumps(tools, separators=(",", ":"))
    q = "Find the current temperature in San Francisco."
    prompt1 = f"TOOLS: {tools_json}\nUSER: {q}\nASSISTANT:"
    dummy = json.dumps({"temperature": 18, "unit": "celsius", "condition": "foggy"}, separators=(",", ":"))
    free_call = constrained_tool_gen(vlm.llm, tok, dev, prompt1, tools_json, GPT2_VALID, EOT, max_new=48)
    # joint7's text path over-abstains; also try FORCING the call start (tools are clearly relevant)
    forced = _free_greedy(vlm.llm, tok, dev, prompt1 + ' [', GPT2_VALID, EOT, 48)
    call = ("[" + forced).split("]")[0] + "]" if "}" in forced else "[" + forced
    prompt2 = f"{prompt1} {call}\nTOOL: {dummy}\nASSISTANT:"
    answer = _free_greedy(vlm.llm, tok, dev, prompt2, GPT2_VALID, EOT, max_new)
    print("\n=== (1) TEXT + TOOLS ===", flush=True)
    print(f"  USER:           {q}", flush=True)
    print(f"  CALL (free):    {free_call}", flush=True)
    print(f"  CALL (forced):  {call}", flush=True)
    print(f"  DUMMY RESULT:   {dummy}", flush=True)
    print(f"  FINAL ANSWER:   {answer}", flush=True)

    # ---- (2) SPEECH AGENT: spoken query + TOOLS -> call -> result -> summary ----
    from transformers import VitsModel, AutoTokenizer
    tts = VitsModel.from_pretrained("facebook/mms-tts-eng").to(dev).eval()
    ttok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    spoken = "What is the current temperature in San Francisco?"
    with torch.no_grad():
        wav = tts(**ttok(spoken, return_tensors="pt").to(dev)).waveform[0].float().cpu().numpy()
    if tts.config.sampling_rate != WHISPER_SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=tts.config.sampling_rate, target_sr=WHISPER_SR)
    sfeats = spk.encode([wav])

    def _banned(prev, k=3):
        if len(prev) < k:
            return []
        seen = {}
        for j in range(len(prev) - k + 1):
            seen.setdefault(tuple(prev[j:j + k - 1]), []).append(prev[j + k - 1])
        return seen.get(tuple(prev[-(k - 1):]), [])

    @torch.no_grad()
    def gen_embeds(ids, feats, n, rep=False):
        """Generate from a sequence whose SPEECH sentinel is expanded to the spoken-query features."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cur, _ = vlm.build_inputs_embeds(torch.tensor([ids], device=dev), speech_features=feats)
            outs = []
            for _ in range(n):
                lg = vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
                lg[:, GPT2_VALID:] = -float("inf")
                if rep and outs:
                    u = torch.tensor(sorted(set(outs)), device=dev); v = lg[0, u]
                    lg[0, u] = torch.where(v > 0, v / 1.3, v * 1.3)
                    for b in _banned(outs):
                        lg[0, b] = -float("inf")
                t = int(lg.argmax(-1).item())
                if t == EOT:
                    break
                outs.append(t)
                cur = torch.cat([cur, vlm.llm.embed(torch.tensor([[t]], device=dev)).to(cur.dtype)], dim=1)
        return tok.decode(outs).strip()

    pre = tok.encode_ordinary(f"TOOLS: {tools_json}\n")
    call_ids = pre + [SPEECH_TOKEN] + tok.encode_ordinary("\nASSISTANT:")
    scall = gen_embeds(call_ids, sfeats, 48, rep=False)
    sum_ids = (pre + [SPEECH_TOKEN]
               + tok.encode_ordinary(f"\nASSISTANT: {scall}\nTOOL: {dummy}\nASSISTANT:"))
    sanswer = gen_embeds(sum_ids, sfeats, max_new, rep=True)
    print("\n=== (2) SPEECH AGENT (spoken query + tools) ===", flush=True)
    print(f"  SPOKEN-Q:    {spoken}", flush=True)
    print(f"  TOOL CALL:   {scall}", flush=True)
    print(f"  DUMMY RESULT:{dummy}", flush=True)
    print(f"  FINAL ANSWER:{sanswer}", flush=True)
    print("\nDEMO DONE", flush=True)


@app.function(image=vlm_image, volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=2 * 60 * 60, secrets=[HF])
def hf_publish(repo_name: str = "moe-omni-500m", include_weights: bool = True, private: bool = True):
    """Create a (private) HF model repo and upload the code + model card + the unified weights, straight
    from Modal (the HF secret token + the checkpoints both live here)."""
    import os
    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"
    print(f"HF user={user} | repo={repo_id} | private={private} | weights={include_weights}", flush=True)
    try:
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as e:
        raise SystemExit(f"create_repo failed (token may be read-only): {e}")
    api.upload_file(path_or_fileobj="/root/moe-lab/HF_MODEL_CARD.md", path_in_repo="README.md", repo_id=repo_id)
    api.upload_folder(repo_id=repo_id, folder_path="/root/moe-lab", path_in_repo="code",
                      ignore_patterns=["__pycache__*", "*.pyc", "*.pyo", ".git*", "HF_MODEL_CARD.md"])
    print("uploaded code + README", flush=True)
    # promote stalled-run ckpts -> canonical model.pt (both runs hung before writing model.pt; each ckpt
    # holds all four state dicts and is the validated no-regression checkpoint).
    import shutil
    # joint12: chained resumes (joint12 -> joint12b -> joint12c) each added ~1600-2000 steps past the random
    # stall. Use the furthest available link as joint12's weights (joint12c ckpt_1600 ~= 5200 eff steps).
    for src in ["/data/runs/500M_vlm_joint12c/ckpt_1600.pt", "/data/runs/500M_vlm_joint12b/ckpt_2000.pt"]:
        if os.path.exists(src):
            shutil.copy(src, "/data/runs/500M_vlm_joint12/model.pt")
            runs_vol.commit()
            print(f"promoted {src} -> joint12/model.pt (chained resume)", flush=True)
            break
    for run, ck in [("500M_vlm_joint9", "ckpt_2800.pt"), ("500M_vlm_joint10", "ckpt_2000.pt"),
                    ("500M_vlm_joint11", "ckpt_4000.pt"), ("500M_vlm_joint12", "ckpt_1600.pt")]:
        d = f"/data/runs/{run}"
        if not os.path.exists(f"{d}/model.pt") and os.path.exists(f"{d}/{ck}"):
            shutil.copy(f"{d}/{ck}", f"{d}/model.pt")
            runs_vol.commit()
            print(f"promoted {run} {ck} -> model.pt", flush=True)
    if include_weights:
        for src, dst in [("/data/runs/500M_vlm_joint12/model.pt", "weights/joint12_model.pt"),
                         ("/data/runs/500M_vlm_joint11/model.pt", "weights/joint11_model.pt"),
                         ("/data/runs/500M_vlm_joint10/model.pt", "weights/joint10_model.pt"),
                         ("/data/runs/500M_vlm_joint9/model.pt", "weights/joint9_model.pt"),
                         ("/data/runs/500M_vlm_joint8/model.pt", "weights/joint8_model.pt"),
                         ("/data/runs/500M_vlm_joint7/model.pt", "weights/joint7_model.pt"),
                         ("/data/runs/500M_vlm_joint6/model.pt", "weights/joint6_model.pt"),
                         ("/data/runs/500M_vlm_joint5/model.pt", "weights/joint5_model.pt"),
                         ("/data/runs/500M_vlm_tools_all/model.pt", "weights/tools_v1_model.pt"),
                         ("/data/runs/500M_ctx2048/model.pt", "weights/backbone_500M_ctx2048.pt")]:
            if os.path.exists(src):
                print(f"uploading {dst} ({os.path.getsize(src)/1e9:.2f} GB)...", flush=True)
                api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo_id)
            else:
                print(f"(skip {src}: not found)", flush=True)
    print(f"PUBLISHED -> https://huggingface.co/{repo_id}", flush=True)
    return {"repo": repo_id}


@app.local_entrypoint()
def main(action: str = "smoke", max_steps: int = 1500, micro: int = 32, lr: float = 1e-3, n: int = 8,
         stage2_run: str = "", tasks: str = "", limit: int = 0, ckpt: str = "model.pt", prompt: str = "",
         backbone: str = "", vision_id: str = ""):
    if action == "download":
        download.remote()
    elif action == "download_sft":
        download_sft.remote()
    elif action == "smoke":
        smoke.remote()
    elif action == "audio_smoke":
        audio_smoke.remote()
    elif action == "audio_stage1":
        train_audio_stage1.remote(max_steps=(max_steps if max_steps != 1500 else 1200),
                                  micro=(micro if micro != 32 else 16), lr=lr)
    elif action == "caption_audio":
        caption_audio.remote(n=n)
    elif action == "speech_smoke":
        speech_smoke.remote()
    elif action == "speech_stage1":
        train_speech_stage1.remote(max_steps=(max_steps if max_steps != 1500 else 1500),
                                   micro=(micro if micro != 32 else 12), lr=lr)
    elif action == "caption_speech":
        caption_speech.remote(n=n)
    elif action == "speech_wer":
        speech_wer.remote(limit=limit, joint_run=(stage2_run if stage2_run.startswith("500M_vlm_joint") else ""),
                          backbone=backbone)
    elif action == "hotpotqa":
        hotpotqa_eval.remote(joint_run=(stage2_run or "500M_vlm_joint12"), backbone=(backbone or "500M_ctx8k"),
                             limit=(limit or 500))
    elif action == "vqa_lenient":
        vqa_lenient.remote(joint_run=(stage2_run or "500M_vlm_joint12"), backbone=(backbone or "500M_ctx8k"),
                           tasks=(tasks or "pope,vqav2,gqa"), limit=(limit or 1000))
    elif action == "speech_sft":
        train_speech_sft.remote(max_steps=(max_steps if max_steps != 1500 else 1200),
                                micro=(micro if micro != 32 else 8), lr=(2e-5 if lr == 1e-3 else lr))
    elif action == "ask_speech":
        ask_speech.remote(n=(n if n != 8 else 6))
    elif action == "joint":
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 1600),
                           micro=(micro if micro != 32 else 4), lr=(2e-5 if lr == 1e-3 else lr))
    elif action == "unified":
        unified.remote(n=(n if n != 8 else 4), joint_run=stage2_run if stage2_run.startswith("500M_vlm_joint") else "")
    elif action == "joint5":
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 2000),
                           micro=(micro if micro != 32 else 4), lr=(2e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint5", speech_run="500M_vlm_speech_stage1")
    elif action == "joint6":
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 2400),
                           micro=(micro if micro != 32 else 4), lr=(2e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint6", speech_run="500M_vlm_speech_stage1", with_tools=True)
    elif action == "joint7":
        # final omni-AGENT: 6 modalities + tools + text-agent (text->call->summarize) + speech-agent (spoken->call->summarize)
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 3200),
                           micro=(micro if micro != 32 else 4), lr=(2e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint7", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True)
    elif action == "joint8":
        # joint7 + TEXT must-call up-weighted (tools x3) so the text path calls by default; speech path unchanged
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 4000),
                           micro=(micro if micro != 32 else 4), lr=(2e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint8", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=3)
    elif action == "joint9":
        # CONTINUE from joint8 (loads all projectors -> no regression) + STREAM nvidia agentic-v2 + VLM-v1,
        # while replaying every existing path. Streams from HF (nothing fully downloaded).
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 6000),
                           micro=(micro if micro != 32 else 4), lr=(1e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint9", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=2,
                           init_joint="500M_vlm_joint8", stream=True)
    elif action == "ocr_smoke":
        ocr_smoke.remote(n=(n if n != 8 else 3), shards=1)
    elif action == "ocr_quality":
        ocr_quality.remote(joint_run=(stage2_run or "500M_vlm_joint10"), ckpt=ckpt, n=(n if n != 8 else 4),
                           backbone=backbone, vision_id=vision_id)
    elif action == "joint10":
        # CONTINUE from joint9 (all projectors -> no regression) + fold in the ocr_4 OCR stream (rendered
        # Wikipedia-text images, pre-staged local tars), while replaying every existing path incl. streams.
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 6000),
                           micro=(micro if micro != 32 else 4), lr=(1e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint10", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=2,
                           init_joint="500M_vlm_joint9", stream=True, stream_ocr=2)
    elif action == "joint11":
        # LIFT joint10 -> 8k context (500M_ctx8k backbone, rope_theta 1e6) + HIGH-RES vision (siglip2 patch16-512,
        # 1024 tok) in ONE run: init all weights/projectors from joint10, replay every path + hardened OCR with
        # full-doc length (ocr_max_len 4096, fits at 8k). LLM adapts theta+seq+vision-res together.
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 6000),
                           micro=(micro if micro != 32 else 2), lr=(1e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint11", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=2,
                           init_joint="500M_vlm_joint10", stream=True, stream_ocr=2,
                           backbone="500M_ctx8k", vision_id="google/siglip2-so400m-patch16-512",
                           ocr_max_len=2048)   # 4096 OOM'd the first OCR step (5120 tok @8k+512px); 2048 proven
    elif action == "joint12":
        # FINAL: continue joint11 (8k ctx + patch16-512) + smoltalk (chat) + mobile-actions (mobile tool-calling)
        # + Aria-UI desktop UI-grounding (≤200k), replay every prior path. NonBlockingLoader guards the DDP hang.
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 6000),
                           micro=(micro if micro != 32 else 2), lr=(1e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint12", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=2,
                           init_joint="500M_vlm_joint11", stream=True, stream_ocr=2,
                           backbone="500M_ctx8k", vision_id="google/siglip2-so400m-patch16-512",
                           ocr_max_len=2048, stream_smol=2, stream_mobile=2, with_aria=True, aria_max=200000)
    elif action == "newdata_smoke":
        newdata_smoke.remote(n=(n if n != 8 else 2))
    elif action == "convert":
        convert_formats.remote(joint_run=(stage2_run or "500M_vlm_joint12"), backbone=(backbone or "500M_ctx8k"))
    elif action == "to_gguf":
        llamacpp_e2e.remote(joint_run=(stage2_run or "500M_vlm_joint12"), prompt=(tasks or "The capital of France is"))
    elif action == "quantize":
        llamacpp_quantize.remote(joint_run=(stage2_run or "500M_vlm_joint12"), qtype=(tasks or "Q4_K_M"))
    elif action == "dump_embeds":
        dump_embeds.remote(joint_run=(stage2_run or "500M_vlm_joint12"))
    elif action == "export_onnx":
        export_onnx.remote(joint_run=(stage2_run or "500M_vlm_joint12"), speech_only=(tasks == "speech"))
    elif action == "export_projectors":
        print(export_projectors.remote(joint_run=(stage2_run or "500M_vlm_joint12")))
    elif action == "mtmd_vision":
        mtmd_vision.remote(joint_run=(stage2_run or "500M_vlm_joint12"),
                           prompt=(tasks or "Describe this image in detail."))
    elif action == "mtmd_audio":
        mtmd_audio.remote(joint_run=(stage2_run or "500M_vlm_joint12"),
                          prompt=(tasks or "What is being said in the audio?"))
    elif action == "speech_parity":
        speech_parity.remote(joint_run=(stage2_run or "500M_vlm_joint12"),
                             prompt=(tasks or "What is being said in the audio?"))
    elif action == "inspect_convert":
        inspect_convert.remote()
    elif action == "audio_mel_debug":
        audio_mel_debug.remote(joint_run=(stage2_run or "500M_vlm_joint12"))
    elif action == "audio_embd_debug":
        audio_embd_debug.remote(joint_run=(stage2_run or "500M_vlm_joint12"))
    elif action == "joint12_resume2":
        # 2nd chained resume: continue from the updated joint12/model.pt (= joint12b ckpt_2000, ~3600 eff steps)
        # toward a fuller schedule; the random stall recurs ~every 2k steps, so each resume adds ~2k more.
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 4400),
                           micro=(micro if micro != 32 else 2), lr=(8e-6 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint12c", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=2,
                           init_joint="500M_vlm_joint12", stream=True, stream_ocr=2,
                           backbone="500M_ctx8k", vision_id="google/siglip2-so400m-patch16-512",
                           ocr_max_len=2048, stream_smol=2, stream_mobile=2, with_aria=True, aria_max=200000)
    elif action == "joint12_resume":
        # CONTINUE joint12 from its best ckpt (ckpt_1600 -> model.pt) toward a fuller schedule. Same 18-path mix;
        # the recurring stall is random (~2-4k steps), so a fresh resume typically adds steps -> take best ckpt.
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 4400),
                           micro=(micro if micro != 32 else 2), lr=(1e-5 if lr == 1e-3 else lr),
                           save_name="500M_vlm_joint12b", speech_run="500M_vlm_speech_stage1",
                           with_tools=True, with_agent=True, tools_rep=2,
                           init_joint="500M_vlm_joint12", stream=True, stream_ocr=2,
                           backbone="500M_ctx8k", vision_id="google/siglip2-so400m-patch16-512",
                           ocr_max_len=2048, stream_smol=2, stream_mobile=2, with_aria=True, aria_max=200000)
    elif action == "unified5":
        unified5.remote(joint_run=(stage2_run or "500M_vlm_joint5"), n=(n if n != 8 else 4))
    elif action == "image_voice":
        image_voice.remote(n=(n if n != 8 else 4))
    elif action == "demo_weather":
        demo_weather.remote(joint_run=(stage2_run or "500M_vlm_joint6"))
    elif action == "agent_demo":
        agent_demo.remote(joint_run=(stage2_run or "500M_vlm_joint8"), query=(tasks or "What's the weather in Tokyo right now?"), ckpt=ckpt, backbone=backbone)
    elif action == "explain_image":
        explain_image.remote(joint_run=(stage2_run or "500M_vlm_joint9"), img=tasks, prompt=prompt, ckpt=ckpt, backbone=backbone, vision_id=vision_id)
    elif action == "prep_speech_tool":
        prep_speech_tool.remote()
    elif action == "publish":
        # --limit 1 -> skip the multi-GB weight re-upload (code + README only)
        hf_publish.remote(repo_name=(tasks or "moe-omni-500m"), include_weights=(limit == 0))
    elif action == "stage1":
        train_stage1.remote(max_steps=max_steps, micro=micro, lr=lr)
    elif action == "stage2":
        train_stage2.remote(max_steps=max_steps, micro=micro, lr=(2e-5 if lr == 1e-3 else lr))
    elif action == "caption":
        caption.remote(n=n, stage2_run=stage2_run)
    elif action == "vlm_eval":
        # lmms-eval harness (POPE/GQA/VQAv2). joint ckpts pass via stage2_run starting with 500M_vlm_joint.
        run = stage2_run or "500M_vlm_stage2"
        joint = run if run.startswith("500M_vlm_joint") else ""
        vlm_eval_run.remote(stage2_run=("500M_vlm_stage2" if joint else run), joint_run=joint,
                            tasks=(tasks or "pope"), limit=limit, backbone=backbone, vision_id=vision_id)
    else:
        raise SystemExit(f"unknown action {action!r} "
                         "(use download|download_sft|smoke|stage1|stage2|caption|vlm_eval)")
