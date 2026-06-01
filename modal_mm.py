"""Modal harness for the multimodal MoE-VLM (image + audio). TinyLLaVA-style.

  # download the LLaVA-Pretrain (LAION-CC-SBU-558K) alignment data to a volume:
  python -m modal run modal_mm.py --action download

  # GPU smoke: real SigLIP2 + 500M_ctx2048 backbone + MoEVLM forward/backward on a synthetic image:
  python -m modal run modal_mm.py --action smoke
"""
import modal

# image with the vision/LLM training deps (separate from the lm-eval image)
vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "transformers>=4.50,<5", "pillow", "numpy",
                 "huggingface-hub", "accelerate", "sentencepiece", "tiktoken",
                 "soundfile", "librosa",
                 "datasets==2.21.0", "pyarrow==17.0.0", "pandas==2.2.2")   # pinned set: audio decode via soundfile
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(".", "/root/moe-lab")
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


@app.function(image=vlm_image, gpu="H100:8", volumes={"/data": runs_vol, "/llava": data_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train_joint(max_steps: int = 1600, micro: int = 4, lr: float = 2e-5, save_name: str = "500M_vlm_joint",
                stage2_run: str = "500M_vlm_stage2", audio_run: str = "500M_vlm_audio_stage1"):
    """Joint image+video+audio SFT on 8x H100 (torchrun): co-train LLM + mm_projector + audio_projector."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "vlm_joint.py",
           "--stage2", f"/data/runs/{stage2_run}/model.pt", "--audio", f"/data/runs/{audio_run}/audio_projector.pt",
           "--json", "/llava/llava_instruct_150k.json", "--zip", "/llava/train2017.zip",
           "--clotho", "CLAPv2/Clotho", "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr)]
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


@app.local_entrypoint()
def main(action: str = "smoke", max_steps: int = 1500, micro: int = 32, lr: float = 1e-3, n: int = 8,
         stage2_run: str = ""):
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
    elif action == "joint":
        train_joint.remote(max_steps=(max_steps if max_steps != 1500 else 1600),
                           micro=(micro if micro != 32 else 4), lr=(2e-5 if lr == 1e-3 else lr))
    elif action == "unified":
        unified.remote(n=(n if n != 8 else 4), joint_run=stage2_run if stage2_run.startswith("500M_vlm_joint") else "")
    elif action == "stage1":
        train_stage1.remote(max_steps=max_steps, micro=micro, lr=lr)
    elif action == "stage2":
        train_stage2.remote(max_steps=max_steps, micro=micro, lr=(2e-5 if lr == 1e-3 else lr))
    elif action == "caption":
        caption.remote(n=n, stage2_run=stage2_run)
    else:
        raise SystemExit(f"unknown action {action!r} "
                         "(use download|download_sft|smoke|stage1|stage2|caption)")
