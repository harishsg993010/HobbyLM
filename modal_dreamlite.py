"""Modal throughput probe for the DreamLite V0-512 in-context latent U-Net.

Measures fwd+bwd images/sec + peak mem at the real V0 spec (400M params, 64x128 two-panel latent),
then extrapolates full-curriculum cost. Image-diffusion analogue of the LLM speed_probe.

nanogpt-style opts (toggleable): channels_last (NHWC -> tensor-core-friendly convs), torch.compile
(op fusion), fused AdamW, bf16, TF32, flash SDPA. channels_last + compile are the big wins for a
conv U-Net. (GB200 is NOT on Modal; B200 is the top tier.)

  python -m modal run modal_dreamlite.py --action probe --gpu H100      # optimized
  python -m modal run modal_dreamlite.py --action probe --gpu B200
  python -m modal run modal_dreamlite.py --action probe --gpu H100 --opt 0   # baseline
"""
import modal

app = modal.App("dreamlite-probe")

_IGNORE = ["*.gguf", "*.bin", "*.pt", "**/*.pt", "*.zip", "*.safetensors", "*.onnx", "hobby-rs/target",
           "checkpoints", "**/.git", "**/__pycache__", "**/*.pyc", "Megatron-Bridge", "data", "runs"]
_base = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy")
img = _base.add_local_dir(".", "/root/moe-lab", ignore=_IGNORE)
# diffusers-enabled image for DC-AE encode/decode + dataset loading (pip BEFORE add_local_dir)
img_diff = (_base.pip_install("diffusers>=0.32.0", "transformers", "datasets", "pillow", "accelerate")
            .add_local_dir(".", "/root/moe-lab", ignore=_IGNORE))

PRICE = {"H100": 3.95, "B200": 6.25, "H200": 4.54}
pilot_vol = modal.Volume.from_name("dreamlite-cache", create_if_missing=True)


@app.function(image=img_diff, gpu="H100", timeout=30 * 60)
def dcae_check(res: int = 256, n: int = 8):
    """Validate DC-AE f32c32 reconstruction at `res` on real edit images (MagicBrush): report latent
    shape + scaling factor + round-trip PSNR, and return a side-by-side PNG (orig top / recon bottom)."""
    import io, base64, torch, numpy as np
    from diffusers import AutoencoderDC
    from datasets import load_dataset
    from PIL import Image
    dev = "cuda"
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.float32).to(dev).eval()
    sf = getattr(ae.config, "scaling_factor", None)
    print(f"DC-AE loaded | scaling_factor={sf}", flush=True)

    ds = load_dataset("osunlp/MagicBrush", split="train", streaming=True)
    imgs = []
    for ex in ds:
        im = ex["source_img"].convert("RGB").resize((res, res), Image.BICUBIC)
        imgs.append(im)
        if len(imgs) >= n:
            break
    x = torch.stack([torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 127.5 - 1 for im in imgs]).to(dev)
    with torch.no_grad():
        z = ae.encode(x).latent
        print(f"latent shape: {tuple(z.shape)}  (expect ({n}, 32, {res//32}, {res//32}))", flush=True)
        rec = ae.decode(z).sample.clamp(-1, 1)
    # PSNR in [0,1] space (correct peak): rescale [-1,1]->[0,1]
    x01, r01 = (x + 1) / 2, (rec + 1) / 2
    mse = ((x01 - r01) ** 2).mean(dim=[1, 2, 3])
    psnr = (-10 * torch.log10(mse)).tolist()
    print(f"round-trip PSNR: {[round(p,1) for p in psnr]}  mean={sum(psnr)/len(psnr):.1f} dB "
          f"(good VAE ~28-33)", flush=True)

    def to_img(t):
        a = ((t.permute(1, 2, 0).cpu().float() + 1) * 127.5).clamp(0, 255).numpy().astype("uint8")
        return Image.fromarray(a)
    grid = Image.new("RGB", (res * n, res * 2))
    for i in range(n):
        grid.paste(to_img(x[i]), (i * res, 0))
        grid.paste(to_img(rec[i]), (i * res, res))
    buf = io.BytesIO(); grid.save(buf, format="PNG")
    return {"scaling_factor": float(sf) if sf else None, "latent_shape": list(z.shape),
            "psnr_mean": round(sum(psnr) / len(psnr), 2), "psnr": [round(p, 1) for p in psnr],
            "png": base64.b64encode(buf.getvalue()).decode()}


@app.function(image=img_diff, gpu="H100", timeout=6 * 60 * 60, volumes={"/cache": pilot_vol})
def cache_data(res: int = 512, q_flickr: int = 6000, q_cc3m: int = 6000,
               q_omni: int = 10000, q_mb: int = 4000):
    """Cache a MIXTURE of datasets to the volume for the unified gen+edit model: flickr30k + CC3M
    (generation, blank source, task=0) and OmniEdit + MagicBrush (editing, source latent + diff/real
    mask, task=1). Encodes DC-AE latents + CLIP text once -> /cache/mix_{res}.pt. Runs on H100 (the
    encode is the bottleneck, not B200-advantaged)."""
    import os, sys, time, torch, numpy as np
    import torch.nn.functional as F
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from diffusers import AutoencoderDC
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    from PIL import Image
    dev = "cuda"
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.bfloat16).to(dev).eval()
    sf = float(ae.config.scaling_factor)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    lat = res // 32
    cpath = f"/cache/mix_{res}.pt"

    def to_t(im):
        return torch.from_numpy(np.array(im.convert("RGB").resize((res, res), Image.BICUBIC))).permute(2, 0, 1).float() / 127.5 - 1

    ZS, ZT, CTX, MK, TASK = [], [], [], [], []

    @torch.no_grad()
    def flush(srcs, tgts, txts, masks, task):
        t_ = torch.stack(tgts).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zt = ae.encode(t_).latent * sf
            if task == 1:
                s = torch.stack(srcs).to(dev)
                zs = ae.encode(s).latent * sf
            else:
                zs = torch.zeros_like(zt)
            ids = tok(txts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids.to(dev)
            ctx = clip(ids).last_hidden_state
        if task == 1 and masks[0] is not None:
            mk = torch.stack([torch.from_numpy(np.array(m.convert("L").resize((lat, lat), Image.BICUBIC)) / 255.0).float()
                              for m in masks]).to(dev)[:, None]
        elif task == 1:
            s = torch.stack(srcs).to(dev)
            d = (s - t_).abs().mean(1, keepdim=True)
            mk = (F.interpolate(d, size=(lat, lat), mode="area") > 0.08).float()
        else:
            mk = torch.zeros(zt.shape[0], 1, lat, lat, device=dev)
        ZS.append(zs.bfloat16()); ZT.append(zt.bfloat16()); CTX.append(ctx.bfloat16())
        MK.append(mk.bfloat16()); TASK.append(torch.full((zt.shape[0],), task, dtype=torch.long))

    specs = [
        # generation (blank source, task=0) — CC3M webdataset (script-free); flickr30k dropped (script-based)
        ("pixparse/cc3m-wds", "train", 0, q_cc3m + q_flickr, lambda ex: (None, ex["jpg"], ex["txt"], None)),
        # editing (source latent + mask, task=1)
        ("TIGER-Lab/OmniEdit-Filtered-1.2M", "train", 1, q_omni,
         lambda ex: (ex["src_img"], ex["edited_img"], (ex["edited_prompt_list"] or ["edit the image"])[0], None)),
        ("osunlp/MagicBrush", "train", 1, q_mb,
         lambda ex: (ex["source_img"], ex["target_img"], ex["instruction"], ex["mask_img"])),
    ]
    sb = 32 if res <= 512 else 8                                # encode sub-batch
    for ds_id, split, task, quota, mp in specs:
        if quota <= 0:
            continue
        t0 = time.time(); cnt = 0
        srcs, tgts, txts, masks = [], [], [], []
        ds = load_dataset(ds_id, split=split, streaming=True)
        for ex in ds:
            try:
                s, t, txt, m = mp(ex)
                tt = to_t(t)
            except Exception:
                continue
            tgts.append(tt); txts.append(txt or "")
            srcs.append(to_t(s) if s is not None else None); masks.append(m)
            if len(tgts) >= 32:
                flush(srcs, tgts, txts, masks, task); cnt += len(tgts); srcs, tgts, txts, masks = [], [], [], []
            if cnt >= quota:
                break
        if tgts:
            flush(srcs, tgts, txts, masks, task); cnt += len(tgts)
        print(f"{ds_id}: {cnt} ({'gen' if task == 0 else 'edit'}) in {time.time()-t0:.0f}s", flush=True)
    ZS = torch.cat(ZS); ZT = torch.cat(ZT); CTX = torch.cat(CTX); MK = torch.cat(MK); TASK = torch.cat(TASK)
    torch.save({"ZS": ZS.cpu(), "ZT": ZT.cpu(), "CTX": CTX.cpu(), "MK": MK.cpu(), "TASK": TASK}, cpath)
    pilot_vol.commit()
    g, e = int((TASK == 0).sum()), int((TASK == 1).sum())
    print(f"CACHED {ZS.shape[0]} total -> {cpath} | gen={g} edit={e}", flush=True)
    return {"total": int(ZS.shape[0]), "gen": g, "edit": e, "path": cpath}


# ---- 20x scaled cache (v2): threaded-prefetch decode + CC12M + token-IDs (not 55GB CTX) + sharded ----
GEN_EDIT_SPECS = [
    # (ds_id, split, task, default_quota, field-mapper)  -- field-mapper returns (src, tgt, txt, mask)
    ("pixparse/cc3m-wds", "train", 0, 140_000, lambda ex: (None, ex["jpg"], ex.get("txt") or "", None)),
    ("pixparse/cc12m-wds", "train", 0, 320_000,
     lambda ex: (None, ex["jpg"], ex.get("txt") or ex.get("caption") or "", None)),
    ("TIGER-Lab/OmniEdit-Filtered-1.2M", "train", 1, 40_000,
     lambda ex: (ex["src_img"], ex["edited_img"], (ex["edited_prompt_list"] or ["edit the image"])[0], None)),
    ("osunlp/MagicBrush", "train", 1, 2_000,
     lambda ex: (ex["source_img"], ex["target_img"], ex["instruction"], ex["mask_img"])),
]


@app.function(image=img_diff, gpu="H100", timeout=6 * 60 * 60, volumes={"/cache": pilot_vol})
def cache_shard(res: int = 512, shard: int = 0, n_shards: int = 4, scale: float = 1.0):
    """One shard of the 20x cache. Stores DC-AE latents + CLIP TOKEN-IDs (recompute CTX at train time
    -> cache ~10GB not ~64GB). Threaded-prefetch decode (16 workers) keeps the GPU encode (the measured
    bottleneck, ~111 img/s H100) fed. `scale` multiplies all quotas (use <1 for smoke tests)."""
    import os, sys, time, torch, numpy as np
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from diffusers import AutoencoderDC
    from transformers import CLIPTokenizer
    from datasets import load_dataset
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True                      # tolerate corrupt/truncated stream images
    dev = "cuda"
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.bfloat16).to(dev).eval()
    sf = float(ae.config.scaling_factor)
    for p in ae.parameters():
        p.requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    lat = res // 32

    def to_arr(im):
        return np.asarray(im.convert("RGB").resize((res, res), Image.BICUBIC), dtype=np.uint8)

    ZS, ZT, IDS, MK, TASK = [], [], [], [], []

    @torch.no_grad()
    def flush(items, task):                                     # items: (s_arr|None, t_arr, txt, mask|None)
        t_ = torch.from_numpy(np.stack([it[1] for it in items])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        s_ = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zt = ae.encode(t_).latent * sf
            if task == 1:
                s_ = torch.from_numpy(np.stack([it[0] for it in items])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
                zs = ae.encode(s_).latent * sf
            else:
                zs = torch.zeros_like(zt)
        ids = tok([it[2] for it in items], padding="max_length", max_length=64,
                  truncation=True, return_tensors="pt").input_ids.int()        # token-IDs, not hidden states
        if task == 1 and items[0][3] is not None:
            mk = torch.stack([torch.from_numpy(np.array(it[3].convert("L").resize((lat, lat), Image.BICUBIC)) / 255.0).float()
                              for it in items]).to(dev)[:, None]
        elif task == 1:
            d = (s_ - t_).abs().mean(1, keepdim=True)
            mk = (F.interpolate(d, size=(lat, lat), mode="area") > 0.08).float()
        else:
            mk = torch.zeros(zt.shape[0], 1, lat, lat, device=dev)
        ZS.append(zs.bfloat16().cpu()); ZT.append(zt.bfloat16().cpu()); IDS.append(ids.cpu())
        MK.append(mk.bfloat16().cpu()); TASK.append(torch.full((zt.shape[0],), task, dtype=torch.long))

    sb = 32 if res <= 512 else 8                                # encode sub-batch
    for ds_id, split, task, base_q, mp in GEN_EDIT_SPECS:
        quota = int(base_q * scale) // n_shards
        if quota <= 0:
            continue
        t0 = time.time(); cnt = 0
        ds = load_dataset(ds_id, split=split, streaming=True)
        try:
            ds = ds.shard(num_shards=n_shards, index=shard)    # file-level shard (efficient on webdataset)
        except Exception:
            pass

        def decode_one(ex):
            try:
                s, t, txt, m = mp(ex)
                return (to_arr(s) if s is not None else None, to_arr(t), txt or "", m)
            except Exception:
                return None

        group = []
        with ThreadPoolExecutor(16) as pool:
            it = iter(ds)
            while cnt < quota:
                try:
                    ex = next(it)                              # decode happens here (HF Image feature)
                except StopIteration:
                    break
                except Exception:
                    continue                                   # skip corrupt/truncated image, keep streaming
                group.append(ex)
                if len(group) >= 64:
                    dec = [d for d in pool.map(decode_one, group) if d is not None]
                    for j in range(0, len(dec), sb):
                        flush(dec[j:j + sb], task)
                    cnt += len(dec); group = []
            if group and cnt < quota:
                dec = [d for d in pool.map(decode_one, group) if d is not None]
                for j in range(0, len(dec), sb):
                    flush(dec[j:j + sb], task)
                cnt += len(dec)
        print(f"  [s{shard}] {ds_id}: {cnt} ({'gen' if task == 0 else 'edit'}) "
              f"in {time.time()-t0:.0f}s ({cnt/max(1,time.time()-t0):.0f} img/s)", flush=True)

    ZS = torch.cat(ZS); ZT = torch.cat(ZT); IDS = torch.cat(IDS); MK = torch.cat(MK); TASK = torch.cat(TASK)
    p = f"/cache/mix_{res}_s{shard}.pt"
    torch.save({"ZS": ZS, "ZT": ZT, "IDS": IDS, "MK": MK, "TASK": TASK}, p)
    pilot_vol.commit()
    g, e = int((TASK == 0).sum()), int((TASK == 1).sum())
    print(f"[s{shard}] wrote {ZS.shape[0]} -> {p} | gen={g} edit={e}", flush=True)
    return {"shard": shard, "total": int(ZS.shape[0]), "gen": g, "edit": e}


@app.function(image=img, timeout=15 * 60, volumes={"/cache": pilot_vol})
def vol_op(op: str = "ls", src: str = "", dst: str = ""):
    """Volume utility: ls / copy. Used to back up the 1.0 model before a retrain overwrites it."""
    import os, shutil
    if op == "copy":
        assert os.path.exists(src), f"{src} missing"
        shutil.copy(src, dst); pilot_vol.commit()
        return {"copied": src, "to": dst, "bytes": os.path.getsize(dst)}
    return {"files": sorted(f"{f} ({os.path.getsize('/cache/'+f)//1_000_000}MB)"
                            for f in os.listdir("/cache"))}


@app.function(image=img_diff, timeout=60 * 60, memory=220000, volumes={"/cache": pilot_vol})
def cache_merge(res: int = 512, prefix: str = "mix"):
    """Concat all {prefix}_{res}_s*.pt shards -> {prefix}_{res}.pt (the file train_pilot loads)."""
    import torch, glob
    parts = sorted(glob.glob(f"/cache/{prefix}_{res}_s*.pt"))
    assert parts, f"no shards {prefix}_{res}_s*.pt found"
    acc = {k: [] for k in ["ZS", "ZT", "IDS", "MK", "TASK"]}
    for p in parts:
        d = torch.load(p, map_location="cpu")
        for k in acc:
            acc[k].append(d[k])
    out = {k: torch.cat(v) for k, v in acc.items()}
    torch.save(out, f"/cache/{prefix}_{res}.pt")
    pilot_vol.commit()
    g, e = int((out["TASK"] == 0).sum()), int((out["TASK"] == 1).sum())
    gb = sum(v.element_size() * v.nelement() for v in out.values()) / 1e9
    print(f"merged {len(parts)} shards -> /cache/{prefix}_{res}.pt | {out['ZS'].shape[0]} "
          f"({g} gen / {e} edit) | {gb:.1f} GB", flush=True)
    return {"total": int(out["ZS"].shape[0]), "gen": g, "edit": e, "size_gb": round(gb, 1)}


# ---- continued-pretrain mixture (non-gated, object + watermark-free): ImageNet + Midjourney + UltraEdit ----
CONT_SPECS = [
    ("evanarlian/imagenet_1k_resized_256", "train", 0, 50_000, "imagenet"),     # clean single objects
    ("Photoroom/midjourney-v6-recap", "train", 0, 1_000_000, "mj"),             # full watermark-free aesthetic
    ("BleachNick/UltraEdit_500k", "FreeForm", 1, 100_000, "ultraedit"),         # broaden edits
    ("TIGER-Lab/OmniEdit-Filtered-1.2M", "train", 1, 100_000, "omniedit"),      # deepen edits (instruction-rich)
]
# 1024px object-accuracy finetune: FLUX-Reason-6M (FLUX.1-dev, literal/accurate, quality-filtered) + a little MJ polish
HUMAN_KW = (" person", "people", " man", "woman", " men ", "women", "child", " kid", " girl", " boy",
            " baby", " lady", " guy", "portrait", " face", "human", "couple", "family", "crowd",
            "athlete", "dancer", "businessman", "businesswoman", "elderly", "teenager", " chef", "worker")
CONT1024_SPECS = [
    ("LucasFang/FLUX-Reason-6M", "train", 0, 500_000, "flux6m"),        # general: keep lifting objects + scenes
    ("LucasFang/FLUX-Reason-6M", "train", 0, 150_000, "flux6m_human"),  # human BOOST (extra people on top)
    ("Photoroom/midjourney-v6-recap", "train", 0, 50_000, "mj"),        # aesthetic polish
]


@app.function(image=img_diff, gpu="H100", timeout=6 * 60 * 60, volumes={"/cache": pilot_vol})
def cache_cont_shard(res: int = 512, shard: int = 0, n_shards: int = 4, scale: float = 1.0):
    """One shard of the continued-pretrain cache (CONT_SPECS). Same latent+token-ID format as cache_shard,
    different sources. ImageNet captions come from the ClassLabel names; Midjourney from the VLM recaptions."""
    import os, sys, time, torch, numpy as np
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from diffusers import AutoencoderDC
    from transformers import CLIPTokenizer
    from datasets import load_dataset
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    dev = "cuda"
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.bfloat16).to(dev).eval()
    sf = float(ae.config.scaling_factor)
    for p in ae.parameters():
        p.requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    lat = res // 32

    def to_arr(im):
        return np.asarray(im.convert("RGB").resize((res, res), Image.BICUBIC), dtype=np.uint8)

    ZS, ZT, IDS, MK, TASK = [], [], [], [], []

    @torch.no_grad()
    def flush(items, task):
        t_ = torch.from_numpy(np.stack([it[1] for it in items])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
        s_ = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zt = ae.encode(t_).latent * sf
            if task == 1:
                s_ = torch.from_numpy(np.stack([it[0] for it in items])).permute(0, 3, 1, 2).to(dev).float() / 127.5 - 1
                zs = ae.encode(s_).latent * sf
            else:
                zs = torch.zeros_like(zt)
        ids = tok([it[2] for it in items], padding="max_length", max_length=64,
                  truncation=True, return_tensors="pt").input_ids.int()
        if task == 1:
            d = (s_ - t_).abs().mean(1, keepdim=True)
            mk = (F.interpolate(d, size=(lat, lat), mode="area") > 0.08).float()
        else:
            mk = torch.zeros(zt.shape[0], 1, lat, lat, device=dev)
        ZS.append(zs.bfloat16().cpu()); ZT.append(zt.bfloat16().cpu()); IDS.append(ids.cpu())
        MK.append(mk.bfloat16().cpu()); TASK.append(torch.full((zt.shape[0],), task, dtype=torch.long))

    specs = CONT1024_SPECS if res >= 1024 else CONT_SPECS       # 1024 = Midjourney-only sharpness set
    sb = 32 if res <= 512 else 8                                # encode sub-batch (1024 = 4x activation)
    for ds_id, split, task, base_q, kind in specs:
        quota = int(base_q * scale) // n_shards
        if quota <= 0:
            continue
        t0 = time.time(); cnt = 0
        ds = load_dataset(ds_id, split=split, streaming=True)
        try:
            ds = ds.shard(num_shards=n_shards, index=shard)
        except Exception:
            pass
        names = None
        if kind == "imagenet":
            try:
                names = ds.features["label"].names
            except Exception:
                names = None

        def decode_one(ex, kind=kind, names=names):
            try:
                if kind == "imagenet":
                    nm = names[ex["label"]].split(",")[0].strip() if names else "object"
                    return (None, to_arr(ex["image"]), f"a photo of a {nm}", None)
                if kind in ("flux6m", "flux6m_human"):          # quality-filtered FLUX.1-dev images
                    if (ex.get("score_image_clarity") or 0) < 7 or (ex.get("score_image_structure") or 0) < 6:
                        return None
                    cap = ex.get("caption_composition") or ex.get("caption_original") or ""
                    if kind == "flux6m_human" and not any(k in cap.lower() for k in HUMAN_KW):
                        return None                             # human-boost spec: require a person in the caption
                    return (None, to_arr(ex["image"]), cap, None)
                if kind == "mj":
                    cap = ex.get("gemini") or ex.get("qwen3") or ex.get("llava") or ex.get("prompt") or ""
                    return (None, to_arr(ex["image"]), cap, None)
                if kind == "omniedit":
                    return (to_arr(ex["src_img"]), to_arr(ex["edited_img"]),
                            (ex.get("edited_prompt_list") or ["edit the image"])[0], None)
                # ultraedit (task 1)
                return (to_arr(ex["source_image"]), to_arr(ex["edited_image"]),
                        ex.get("edit_prompt") or "edit the image", None)
            except Exception:
                return None

        group = []
        with ThreadPoolExecutor(16) as pool:
            it = iter(ds)
            while cnt < quota:
                try:
                    ex = next(it)
                except StopIteration:
                    break
                except Exception:
                    continue
                group.append(ex)
                if len(group) >= 64:
                    dec = [d for d in pool.map(decode_one, group) if d is not None]
                    for j in range(0, len(dec), sb):
                        flush(dec[j:j + sb], task)
                    cnt += len(dec); group = []
            if group and cnt < quota:
                dec = [d for d in pool.map(decode_one, group) if d is not None]
                for j in range(0, len(dec), sb):
                    flush(dec[j:j + sb], task)
                cnt += len(dec)
        print(f"  [s{shard}] {ds_id}: {cnt} ({'gen' if task == 0 else 'edit'}) "
              f"in {time.time()-t0:.0f}s ({cnt/max(1,time.time()-t0):.0f} img/s)", flush=True)

    ZS = torch.cat(ZS); ZT = torch.cat(ZT); IDS = torch.cat(IDS); MK = torch.cat(MK); TASK = torch.cat(TASK)
    p = f"/cache/mix_cont_{res}_s{shard}.pt"
    torch.save({"ZS": ZS, "ZT": ZT, "IDS": IDS, "MK": MK, "TASK": TASK}, p)
    pilot_vol.commit()
    g, e = int((TASK == 0).sum()), int((TASK == 1).sum())
    print(f"[s{shard}] wrote {ZS.shape[0]} -> {p} | gen={g} edit={e}", flush=True)
    return {"shard": shard, "total": int(ZS.shape[0]), "gen": g, "edit": e}


@app.function(image=img_diff, gpu="B200", timeout=6 * 60 * 60, volumes={"/cache": pilot_vol})
def train_pilot(steps: int = 1500, res: int = 512, micro: int = 32, lr: float = 5e-5,
                cfg_scale: float = 3.0, alpha_fg: float = 1.5, cache: str = "mix",
                resume: str = "", replay: float = 0.0, out: str = "", save_every: int = 0):
    """256/512px DC-AE DiT pilot: two-panel [noisy_target | source] flow-matching on MagicBrush edit
    pairs, conditioned on frozen CLIP text (a stand-in for the 500M VLM to isolate the generator).
    Proves the in-context editing mechanism. Returns loss curve + before/after sample PNG."""
    import io, base64, time, torch, numpy as np
    import torch.nn.functional as F
    import os, sys
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from diffusers import AutoencoderDC
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    from PIL import Image
    from dreamlite.dit import DreamLiteDiT, DiTConfig, count_params
    dev = "cuda"

    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.bfloat16).to(dev).eval()
    sf = float(ae.config.scaling_factor)
    for p in ae.parameters():
        p.requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    for p in clip.parameters():
        p.requires_grad_(False)

    lat = res // 32
    cfg = DiTConfig(in_channels=34, out_channels=32, latent_h=lat, panel_w=lat, patch=1,
                    d_model=1024, depth=16, heads=16, mlp_ratio=3.0, ctx_dim=768)  # CLIP-L = 768
    model = DreamLiteDiT(cfg).to(dev)
    if resume:                                                 # continued pretraining: warm-start from a checkpoint
        ck = torch.load(resume, map_location=dev)
        sd = dict(ck["sd"])
        if "pos" in sd and sd["pos"].shape != model.pos.shape:  # res change -> interpolate the learned pos-embed
            old = sd["pos"]; No = old.shape[1]; dd = old.shape[2]
            oh = int(round((No / 2) ** 0.5)); ow = No // oh     # canvas is 2:1 (width = 2*height)
            nh = cfg.latent_h // cfg.patch; nw = (2 * cfg.panel_w) // cfg.patch
            g = old.reshape(1, oh, ow, dd).permute(0, 3, 1, 2).float()
            g = F.interpolate(g, size=(nh, nw), mode="bicubic", align_corners=False)
            sd["pos"] = g.permute(0, 2, 3, 1).reshape(1, nh * nw, dd).to(old.dtype)
            print(f"interpolated pos-embed {tuple(old.shape)} ({oh}x{ow}) -> {tuple(sd['pos'].shape)} ({nh}x{nw})", flush=True)
        model.load_state_dict(sd)
        print(f"RESUMED from {resume} (trained {ck.get('steps')} steps)", flush=True)
    run = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.02, fused=True)
    ema = [p.detach().clone() for p in model.parameters()]     # EMA weights (smooth metastable jumps)
    warmup = 500 if resume else 1000                           # shorter warmup when resuming
    print(f"pilot: DiT {count_params(model)/1e6:.0f}M, {model.n_tokens} tok, res={res}, micro={micro} | "
          f"resume={bool(resume)} replay={replay}", flush=True)

    def to_t(im):
        return torch.from_numpy(np.array(im.convert("RGB").resize((res, res), Image.BICUBIC))).permute(2, 0, 1).float() / 127.5 - 1

    # ---- load the MIXTURE cache (cache_shard v2: latents + CLIP token-IDs; CTX recomputed per-batch) ----
    cpath = f"/cache/{cache}_{res}.pt"
    assert os.path.exists(cpath), f"{cpath} missing — run the cache step first"
    d = torch.load(cpath, map_location=dev)
    ZS, ZT, MK, TASK = d["ZS"].to(dev), d["ZT"].to(dev), d["MK"].to(dev), d["TASK"].to(dev)
    IDS = d["IDS"].to(dev).long()                              # CLIP token-IDs -> CTX computed on the fly
    # sanitize cached latents IN-PLACE (avoid full copies -> OOM on large caches) + clamp outliers
    ZS.nan_to_num_().clamp_(-6, 6); ZT.nan_to_num_().clamp_(-6, 6)
    # EMPIRICAL unit-variance normalization (don't trust DC-AE's scaling_factor) — the stability fix
    lat_std = ZT.std().clamp_min(1e-3)
    ZS.div_(lat_std); ZT.div_(lat_std)
    N = ZS.shape[0]

    # ---- edit-only replay from the old pool (retain editing without re-teaching watermarked gen) ----
    RZS = RZT = RMK = RIDS = None; NR = 0
    if replay > 0 and os.path.exists(f"/cache/mix_{res}.pt"):
        r = torch.load(f"/cache/mix_{res}.pt", map_location=dev)
        em = (r["TASK"].to(dev) == 1)                          # edits only
        RZS = (torch.nan_to_num(r["ZS"][em].to(dev)).clamp(-6, 6) / lat_std)
        RZT = (torch.nan_to_num(r["ZT"][em].to(dev)).clamp(-6, 6) / lat_std)
        RMK = r["MK"][em].to(dev); RIDS = r["IDS"][em].to(dev).long(); NR = RZS.shape[0]

    @torch.no_grad()
    def ctx_of(ids):                                          # frozen CLIP text features for a batch of IDs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return clip(ids).last_hidden_state.float()
    print(f"loaded {cache} cache: {N} ({int((TASK==0).sum())} gen / {int((TASK==1).sum())} edit) | "
          f"{tuple(ZS.shape)} | lat_std={lat_std.item():.3f} | edit-replay pool={NR}", flush=True)

    import math
    def lr_at(s):                                              # warmup then cosine to 10%
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    losses = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        rb = int(round(micro * replay)) if NR > 0 else 0       # replay sub-batch (edit-only)
        nb = micro - rb
        idx = torch.randint(0, N, (nb,), device=dev)
        zs, zt, mk, ids, task = ZS[idx], ZT[idx], MK[idx], IDS[idx], TASK[idx]
        if rb > 0:
            ridx = torch.randint(0, NR, (rb,), device=dev)
            zs = torch.cat([zs, RZS[ridx]]); zt = torch.cat([zt, RZT[ridx]])
            mk = torch.cat([mk, RMK[ridx].float()]); ids = torch.cat([ids, RIDS[ridx]])
            task = torch.cat([task, torch.ones(rb, dtype=task.dtype, device=dev)])
        zs, zt, mk = zs.float(), zt.float(), mk.float()
        ctx = ctx_of(ids)                                      # CLIP text features (recomputed, not cached)
        B = micro
        noise = torch.randn_like(zt)
        tt = torch.sigmoid(torch.randn(B, device=dev))         # logit-normal t (SD3): avoids low-noise pathology
        z_t = (1 - tt)[:, None, None, None] * noise + tt[:, None, None, None] * zt
        v_tgt = zt - noise                                      # rectified-flow velocity
        # mask-dropout: 50% of samples get ZERO masks so no-mask inference is in-distribution
        keepm = (torch.rand(B, 1, 1, 1, device=dev) < 0.5).float()
        mki = mk * keepm
        latc = torch.cat([z_t, zs], dim=-1)                    # (B,32,lat,2lat)
        em = torch.cat([mki, torch.zeros_like(mki)], dim=-1)   # edit mask: left only
        pm = torch.cat([(1 - mk) * keepm, torch.zeros_like(mki)], dim=-1)
        inp = torch.cat([latc, em, pm], dim=1)                 # (B,34,lat,2lat)
        # CFG: drop the instruction 10% of the time (null = zeros) so guided sampling works
        ctxd = ctx * (torch.rand(B, 1, 1, device=dev) >= 0.1).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = run(inp, tt, ctxd, task)[..., :latc.shape[-1] // 2]   # left (target) panel
            w = 1 + alpha_fg * mk                               # foreground-weighted (true mask)
            psl = (w * (v.float() - v_tgt) ** 2).mean(dim=[1, 2, 3])   # per-sample
            loss = psl.clamp(max=4.0).mean()                    # cap outliers -> no metastable jumps
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):                               # skip the rare bad batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():                              # EMA update
                for e, p in zip(ema, model.parameters()):
                    e.mul_(0.999).add_(p.detach(), alpha=0.001)
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {(time.time()-t0)/(step+1)*1000:.0f}ms/step", flush=True)
        if save_every and step > 0 and step % save_every == 0:  # periodic EMA checkpoint (survive preemption)
            mname = out or f"model_{res}"
            bak = [p.detach().clone() for p in model.parameters()]
            with torch.no_grad():
                for p, e in zip(model.parameters(), ema):
                    p.copy_(e)
                torch.save({"sd": model.state_dict(), "cfg_dict": cfg.__dict__, "lat_std": float(lat_std),
                            "sf": sf, "steps": step}, f"/cache/{mname}.pt")
                for p, b in zip(model.parameters(), bak):
                    p.copy_(b)
            pilot_vol.commit()
            print(f"  [ckpt] saved EMA @ step {step} -> /cache/{mname}.pt", flush=True)

    # load EMA weights into the model (smoother than raw) for sampling + the saved checkpoint
    with torch.no_grad():
        for p, e in zip(model.parameters(), ema):
            p.copy_(e)
    mname = out or f"model_{res}"
    torch.save({"sd": model.state_dict(), "cfg_dict": cfg.__dict__,
                "lat_std": float(lat_std), "sf": sf, "steps": steps}, f"/cache/{mname}.pt")
    pilot_vol.commit()
    print(f"saved EMA model -> /cache/{mname}.pt", flush=True)

    # --- sample (CFG): generate (blank source, task=0) or edit (source latent, task=1) ---
    @torch.no_grad()
    def sample(zs, ctx, task, n_steps=50, cfg=cfg_scale):
        n = zs.shape[0]
        z = torch.randn(n, 32, lat, lat, device=dev)
        em = torch.zeros(n, 1, lat, 2 * lat, device=dev)            # zero masks (in-distribution)
        null = torch.zeros_like(ctx)
        for i in range(n_steps):
            tt = torch.full((n,), i / n_steps, device=dev)
            inp = torch.cat([torch.cat([z, zs], dim=-1), em, em], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vc = run(inp, tt, ctx, task)[..., :lat].float()
                vu = run(inp, tt, null, task)[..., :lat].float()
            z = z + (vu + cfg * (vc - vu)) / n_steps               # classifier-free guidance
        return z

    # pick 2 generation + 2 edit examples for the grid
    gi = (TASK == 0).nonzero(as_tuple=True)[0][:2]
    ei = (TASK == 1).nonzero(as_tuple=True)[0][:2]
    idx = torch.cat([gi, ei])
    zs, zt, task = ZS[idx].float(), ZT[idx].float(), TASK[idx]
    ctx = ctx_of(IDS[idx])                                     # CLIP text features for the sample grid
    out = sample(zs, ctx, task)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        src_img = ae.decode((zs * lat_std / sf).bfloat16()).sample.float().clamp(-1, 1)   # blank for gen
        tgt_img = ae.decode((zt * lat_std / sf).bfloat16()).sample.float().clamp(-1, 1)
        gen_img = ae.decode((out * lat_std / sf).bfloat16()).sample.float().clamp(-1, 1)

    def to_img(t):
        a = ((t.permute(1, 2, 0).cpu().float() + 1) * 127.5).clamp(0, 255).numpy().astype("uint8")
        return Image.fromarray(a)
    n = idx.shape[0]
    grid = Image.new("RGB", (res * n, res * 3))
    for i in range(n):                                           # cols 0-1 = GENERATE, 2-3 = EDIT
        grid.paste(to_img(src_img[i]), (i * res, 0))            # source (blank for gen)
        grid.paste(to_img(gen_img[i]), (i * res, res))         # MODEL output
        grid.paste(to_img(tgt_img[i]), (i * res, res * 2))     # ground-truth target
    buf = io.BytesIO(); grid.save(buf, format="PNG")
    print(f"final loss (last 50 avg): {sum(losses[-50:])/min(50,len(losses)):.4f}", flush=True)
    return {"loss_first": round(sum(losses[:50]) / 50, 4),
            "loss_last": round(sum(losses[-50:]) / min(50, len(losses)), 4),
            "steps": steps, "png": base64.b64encode(buf.getvalue()).decode()}


GEN_PROMPTS = [
    "a golden retriever puppy playing in a field of wildflowers",
    "a futuristic city skyline at sunset, neon lights",
    "a steaming cup of coffee on a rustic wooden table",
    "a snow-capped mountain reflected in a calm lake",
    "a red sports car on a coastal highway",
    "a cozy library with tall bookshelves and warm light",
    "a tropical beach with palm trees and turquoise water",
    "a bowl of fresh strawberries on a kitchen counter",
    "a city street in the rain at night, reflections",
]

OBJECT_PROMPTS = [
    "a single white ceramic coffee cup, centered, plain background, product photo, sharp focus",
    "a red apple on a white table, studio lighting, sharp",
    "a pair of brown leather boots, product photography, white background",
    "a yellow school bus on a city street",
    "a vintage film camera on a wooden surface, close up",
    "a green ceramic teapot, plain white background, studio photo",
    "a slice of chocolate cake on a white plate",
    "a blue bicycle leaning against a brick wall",
    "a glass bottle of orange juice, studio product shot",
]
HUMAN_PROMPTS = [
    "a professional headshot portrait of a woman, studio lighting, sharp focus",
    "a man in a suit standing in a city street, full body",
    "a young child playing in a park, sunny day",
    "an elderly man with glasses reading a book, warm light",
    "a woman chef cooking in a kitchen, candid",
    "a athlete running on a track, dynamic motion",
    "a girl with curly hair smiling, close up portrait",
    "two friends laughing together at a cafe",
    "a businesswoman walking confidently, full body",
]
NEG_DEFAULT = "blurry, low quality, watermark, signature, text, jpeg artifacts, deformed, distorted"


@app.function(image=img_diff, gpu="H100", timeout=20 * 60, volumes={"/cache": pilot_vol})
def sample_gen(prompts: str = "", n_steps: int = 100, cfg: float = 5.0, res: int = 512, cols: int = 3,
               neg: str = NEG_DEFAULT, mdl: str = ""):
    """Text->image generation from the trained model_{res}.pt: blank source panel, CFG flow-matching,
    decode the left panel through sana-1.1. Returns a labeled grid PNG. `prompts` is '|'-separated
    (or 'objects'/'scenes' for the built-in sets). `neg` = negative prompt for the CFG uncond branch
    (steers away from blur/watermarks); empty -> zero-embedding uncond (training-calibrated)."""
    import os, sys, io, base64, torch, numpy as np
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from diffusers import AutoencoderDC
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image, ImageDraw
    from dreamlite.dit import DreamLiteDiT, DiTConfig
    dev = "cuda"
    ckpt = torch.load(f"/cache/{mdl or f'model_{res}'}.pt", map_location=dev)
    lat_std = ckpt["lat_std"]; sf = ckpt["sf"]
    model = DreamLiteDiT(DiTConfig(**ckpt["cfg_dict"])).to(dev).eval()
    model.load_state_dict(ckpt["sd"])
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.bfloat16).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    lat = res // 32
    if prompts.strip() == "objects":
        plist = OBJECT_PROMPTS
    elif prompts.strip() == "humans":
        plist = HUMAN_PROMPTS
    elif prompts.strip() == "scenes" or not prompts.strip():
        plist = GEN_PROMPTS
    else:
        plist = [p.strip() for p in prompts.split("|") if p.strip()]
    n = len(plist)
    print(f"generating {n} images | steps={n_steps} cfg={cfg} | neg={'on' if neg else 'off'} | "
          f"trained {ckpt.get('steps')} steps", flush=True)

    def encode(texts):
        ids = tok(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return clip(ids).last_hidden_state.float()
    ctx = encode(plist)
    uncond = encode([neg] * n) if neg else torch.zeros_like(ctx)   # negative prompt OR zero-embedding
    task = torch.zeros(n, dtype=torch.long, device=dev)
    z = torch.randn(n, 32, lat, lat, device=dev)
    zs = torch.zeros(n, 32, lat, lat, device=dev)
    em = torch.zeros(n, 1, lat, 2 * lat, device=dev)
    with torch.no_grad():
        for i in range(n_steps):                                # CFG flow-matching sampler (Euler)
            tt = torch.full((n,), i / n_steps, device=dev)
            inp = torch.cat([torch.cat([z, zs], dim=-1), em, em], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vc = model(inp, tt, ctx, task)[..., :lat].float()
                vu = model(inp, tt, uncond, task)[..., :lat].float()
            z = z + (vu + cfg * (vc - vu)) / n_steps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            imgs = ae.decode((z * lat_std / sf).bfloat16()).sample.float().clamp(-1, 1)

    def to_img(t):
        a = ((t.permute(1, 2, 0).cpu().float() + 1) * 127.5).clamp(0, 255).numpy().astype("uint8")
        return Image.fromarray(a)
    rows = (n + cols - 1) // cols
    caph = 30
    grid = Image.new("RGB", (res * cols, (res + caph) * rows), (18, 18, 18))
    drw = ImageDraw.Draw(grid)
    for i, p in enumerate(plist):
        r, c = divmod(i, cols)
        grid.paste(to_img(imgs[i]), (c * res, r * (res + caph)))
        drw.text((c * res + 6, r * (res + caph) + res + 8), p[:70], fill=(235, 235, 235))
    buf = io.BytesIO(); grid.save(buf, format="PNG")
    return {"n": n, "cfg": cfg, "png": base64.b64encode(buf.getvalue()).decode()}


@app.function(image=img_diff, gpu="H100", timeout=20 * 60, volumes={"/cache": pilot_vol})
def sample_edit(n: int = 6, n_steps: int = 50, cfg: float = 2.5, res: int = 512,
                mdl: str = "model_cont_512", ds_id: str = "osunlp/MagicBrush", split: str = "dev"):
    """Image EDITING from the trained model: stream held-out (source, instruction, target) triples,
    encode source -> two-panel [noisy_target | source], CFG flow-matching (task=1), decode the left panel.
    Grid rows: source / MODEL edit / ground-truth target. Lower CFG than gen (edits over-fire at high CFG)."""
    import os, sys, io, base64, torch, numpy as np
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from diffusers import AutoencoderDC
    from transformers import CLIPTextModel, CLIPTokenizer
    from datasets import load_dataset
    from PIL import Image, ImageDraw, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from dreamlite.dit import DreamLiteDiT, DiTConfig
    dev = "cuda"
    ckpt = torch.load(f"/cache/{mdl}.pt", map_location=dev)
    lat_std = ckpt["lat_std"]; sf = ckpt["sf"]; lat = res // 32
    model = DreamLiteDiT(DiTConfig(**ckpt["cfg_dict"])).to(dev).eval()
    model.load_state_dict(ckpt["sd"])
    ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                                       torch_dtype=torch.bfloat16).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                         torch_dtype=torch.bfloat16).to(dev).eval()
    print(f"editing {n} held-out pairs from {ds_id}:{split} | cfg={cfg} | {mdl}", flush=True)

    def to_t(im):
        return torch.from_numpy(np.array(im.convert("RGB").resize((res, res), Image.BICUBIC))).permute(2, 0, 1).float() / 127.5 - 1
    srcs, tgts, instrs = [], [], []
    for ex in load_dataset(ds_id, split=split, streaming=True):
        try:
            s, t, ins = ex["source_img"], ex["target_img"], ex["instruction"]
            srcs.append(to_t(s)); tgts.append(to_t(t)); instrs.append(ins or "edit the image")
        except Exception:
            continue
        if len(srcs) >= n:
            break
    n = len(srcs)
    src = torch.stack(srcs).to(dev); tgt = torch.stack(tgts).to(dev)
    ids = tok(instrs, padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = clip(ids).last_hidden_state.float()
        zs = (ae.encode(src).latent * sf) / lat_std                # source latent (normalized like training)
    null = torch.zeros_like(ctx)
    task = torch.ones(n, dtype=torch.long, device=dev)             # edit
    z = torch.randn(n, 32, lat, lat, device=dev)
    em = torch.zeros(n, 1, lat, 2 * lat, device=dev)
    with torch.no_grad():
        for i in range(n_steps):
            tt = torch.full((n,), i / n_steps, device=dev)
            inp = torch.cat([torch.cat([z, zs], dim=-1), em, em], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vc = model(inp, tt, ctx, task)[..., :lat].float()
                vu = model(inp, tt, null, task)[..., :lat].float()
            z = z + (vu + cfg * (vc - vu)) / n_steps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ae.decode((z * lat_std / sf).bfloat16()).sample.float().clamp(-1, 1)

    def to_img(t):
        a = ((t.permute(1, 2, 0).cpu().float() + 1) * 127.5).clamp(0, 255).numpy().astype("uint8")
        return Image.fromarray(a)
    caph = 30
    grid = Image.new("RGB", (res * n, res * 3 + caph), (18, 18, 18))
    drw = ImageDraw.Draw(grid)
    for i in range(n):
        grid.paste(to_img(src[i]), (i * res, 0))                   # row0 source
        grid.paste(to_img(out[i]), (i * res, res))                 # row1 MODEL edit
        grid.paste(to_img(tgt[i]), (i * res, res * 2))             # row2 ground-truth target
        drw.text((i * res + 6, res * 3 + 6), instrs[i][:70], fill=(235, 235, 235))
    buf = io.BytesIO(); grid.save(buf, format="PNG")
    return {"n": n, "cfg": cfg, "png": base64.b64encode(buf.getvalue()).decode()}


def _cache_probe_body(gpu: str, res: int = 512, K: int = 512, lite: int = 0):
    """Find the caching bottleneck: time NETWORK (stream pull), DECODE (jpeg+resize, serial vs
    threaded), and GPU ENCODE (DC-AE, batch sweep) separately. The achievable end-to-end rate is
    min(overlapped_io, best_encode). Prints $/1k-pairs so we can pick the cheapest GPU+config.
    If I/O-bound, a cheap L4 ties an H100 at ~5x lower $/hr; if encode-bound, bigger batch / lite VAE win."""
    import os, time, numpy as np
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from diffusers import AutoencoderDC
    from datasets import load_dataset
    from PIL import Image
    dev = "cuda"
    vae_id = ("mit-han-lab/dc-ae-lite-f32c32-sana-1.1-diffusers" if lite
              else "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers")
    ae = AutoencoderDC.from_pretrained(vae_id, torch_dtype=torch.bfloat16).to(dev).eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    price = {"H100": 3.95, "L4": 0.80, "A10G": 1.10, "A100": 2.10, "B200": 6.25}.get(gpu, 3.95)
    print(f"[{gpu}] VAE={'lite' if lite else 'full'}-1.1 res={res} K={K} ${price}/hr\n", flush=True)

    def to_arr(im):
        return np.asarray(im.convert("RGB").resize((res, res), Image.BICUBIC), dtype=np.uint8)

    # ---- STAGE 1: NETWORK — pull K PIL images (download + HF jpeg-decode on access) ----
    it = iter(load_dataset("pixparse/cc12m-wds", split="train", streaming=True))
    pil = next(it)["jpg"]; pil.convert("RGB")                      # warm connection (excluded)
    t0 = time.time(); pils = [next(it)["jpg"] for _ in range(K)]
    net = K / (time.time() - t0)

    # ---- STAGE 2: DECODE — resize+to-array, serial vs threaded (PIL releases the GIL) ----
    t0 = time.time(); [to_arr(p) for p in pils]
    dec_serial = K / (time.time() - t0)
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        arrs = list(ex.map(to_arr, pils))
    dec_thread = K / (time.time() - t0)
    overlapped = min(net, dec_thread)                             # prefetch hides decode under network

    # ---- STAGE 3: ENCODE — DC-AE forward, batch sweep (pure GPU, synchronized) ----
    # 512px encode is memory-heavy (early stages run at full res) -> small batches, OOM-guarded.
    x = (torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float() / 127.5 - 1)
    enc = {}
    for B in [b for b in (8, 16, 32, 48, 64) if b <= K]:
        try:
            xb = x[:B].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                ae.encode(xb)                                    # warm
            torch.cuda.synchronize(); t0 = time.time(); reps = 10
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(reps):
                    ae.encode(xb).latent
            torch.cuda.synchronize()
            enc[B] = B * reps / (time.time() - t0)
            del xb
        except torch.cuda.OutOfMemoryError:
            print(f"  encode B={B} OOM", flush=True)
        torch.cuda.empty_cache()
    enc_best = max(enc.values()) if enc else 1.0

    e2e = min(overlapped, enc_best)
    cost_1k = 1000 / e2e / 3600 * price
    print(f"  NETWORK  : {net:7.1f} img/s   (stream pull, single iterator)", flush=True)
    print(f"  DECODE   : {dec_serial:7.1f} serial -> {dec_thread:7.1f} threaded(16) img/s", flush=True)
    print(f"  ENCODE   : " + " ".join(f"B{B}={r:.0f}" for B, r in enc.items()) + " img/s", flush=True)
    print(f"  -> overlapped IO {overlapped:.0f} | encode {enc_best:.0f} | E2E {e2e:.0f} img/s", flush=True)
    bn = "NETWORK" if overlapped <= enc_best and net <= dec_thread else ("DECODE" if overlapped <= enc_best else "ENCODE")
    print(f"  -> BOTTLENECK = {bn} | ${cost_1k:.3f}/1k pairs | 562k pairs ~${cost_1k*562:.0f}", flush=True)
    return {"gpu": gpu, "lite": bool(lite), "net": net, "dec_serial": dec_serial,
            "dec_thread": dec_thread, "enc": enc, "e2e": e2e, "bottleneck": bn,
            "cost_per_1k": round(cost_1k, 3), "cost_562k": round(cost_1k * 562)}


@app.function(image=img_diff, gpu="H100", timeout=20 * 60)
def cache_probe_h100(res: int = 512, K: int = 512, lite: int = 0):
    return _cache_probe_body("H100", res, K, lite)


@app.function(image=img_diff, gpu="L4", timeout=20 * 60)
def cache_probe_l4(res: int = 512, K: int = 512, lite: int = 0):
    return _cache_probe_body("L4", res, K, lite)


@app.function(image=img_diff, gpu="A10G", timeout=20 * 60)
def cache_probe_a10g(res: int = 512, K: int = 512, lite: int = 0):
    return _cache_probe_body("A10G", res, K, lite)


def _probe_body(gpu: str, batches, steps, opt, ckpt):
    import os, sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from dreamlite.unet import DreamLiteUNet, V0_512, count_params

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = V0_512
    model = DreamLiteUNet(cfg).to(dev)
    chlast = bool(opt)
    if chlast:
        model = model.to(memory_format=torch.channels_last)
    if ckpt:
        import torch.utils.checkpoint as cp
        for stage in list(model.down_attn) + list(model.up_attn) + [model.mid_attn]:
            for blk in stage:
                fwd = blk.forward
                blk.forward = (lambda f: lambda *a, **k: cp.checkpoint(f, *a, use_reentrant=False, **k))(fwd)
    n = count_params(model)
    run = torch.compile(model) if opt else model
    opt_obj = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=bool(opt))
    print(f"DreamLite V0-512: {n/1e6:.1f}M params | {gpu} bf16 | opt={bool(opt)} "
          f"(compile+channels_last+fused) ckpt={bool(ckpt)}\n", flush=True)

    rows = []
    for B in [int(b) for b in batches.split(",")]:
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(B, cfg.in_channels, 64, 128, device=dev)
            vtgt = torch.randn(B, cfg.out_channels, 64, 128, device=dev)
            if chlast:
                x = x.to(memory_format=torch.channels_last)
                vtgt = vtgt.to(memory_format=torch.channels_last)
            t = torch.rand(B, device=dev)
            ctx = torch.randn(B, 256, cfg.ctx_dim, device=dev)
            task = torch.zeros(B, dtype=torch.long, device=dev)

            def one():
                opt_obj.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = run(x, t, ctx, task)
                    loss = torch.nn.functional.mse_loss(v.float(), vtgt)
                loss.backward()
                opt_obj.step()

            for _ in range(8 if opt else 4):    # extra warmup for compile
                one()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(steps):
                one()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / steps
            ips = B / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            rows.append((B, dt * 1000, ips, peak))
            print(f"  B={B:3d} | {dt*1000:7.1f} ms/step | {ips:7.1f} img/s | peak {peak:5.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B:3d} | OOM", flush=True)
            torch.cuda.empty_cache()
    if not rows:
        return {"error": "all OOM"}

    ips1 = max(r[2] for r in rows)
    ips8 = ips1 * 8 * 0.9
    price = PRICE.get(gpu, 6.25)
    print(f"\nBest single-{gpu}: {ips1:.0f} img/s -> 8x{gpu} ~{ips8:.0f} img/s @ ${price*8:.0f}/hr", flush=True)
    print(f"\n=== cost estimate (8x{gpu}, sample-views = batch x steps) ===", flush=True)
    print(f"{'sample-views':>13} | {'hours':>7} | {'cost $':>8}", flush=True)
    for views in (100e6, 200e6, 300e6):
        hr = views / ips8 / 3600
        print(f"{views/1e6:>10.0f} M | {hr:>7.1f} | {hr*price*8:>8.0f}", flush=True)
    print("(+ VAE ~$100-200, caching ~$20, distillation ~$100; 768/1024 finetunes 2-4x the 512 rate)", flush=True)
    return {"gpu": gpu, "params_M": n / 1e6, "best_img_s": ips1, "ips_8gpu": ips8, "rows": rows}


def _dit_body(gpu: str, batches, steps, res):
    import os, sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from dreamlite.dit import DreamLiteDiT, V0_DCAE_512, V0_DCAE_256, count_params
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = V0_DCAE_256 if int(res) == 256 else V0_DCAE_512
    model = DreamLiteDiT(cfg).to(dev)
    run = torch.compile(model)
    opt_obj = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=True)
    n = count_params(model)
    print(f"DreamLite DiT (DC-AE f32, {res}px): {n/1e6:.1f}M params | {model.n_tokens} tokens | "
          f"{gpu} bf16 opt\n", flush=True)
    rows = []
    for B in [int(b) for b in batches.split(",")]:
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(B, cfg.in_channels, cfg.latent_h, 2 * cfg.panel_w, device=dev)
            vtgt = torch.randn(B, cfg.out_channels, cfg.latent_h, 2 * cfg.panel_w, device=dev)
            t = torch.rand(B, device=dev)
            ctx = torch.randn(B, 256, cfg.ctx_dim, device=dev)
            task = torch.zeros(B, dtype=torch.long, device=dev)

            def one():
                opt_obj.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = run(x, t, ctx, task)
                    loss = torch.nn.functional.mse_loss(v.float(), vtgt)
                loss.backward(); opt_obj.step()

            for _ in range(8):
                one()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(steps):
                one()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / steps
            ips = B / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            rows.append((B, dt * 1000, ips, peak))
            print(f"  B={B:5d} | {dt*1000:7.1f} ms/step | {ips:8.0f} img/s | peak {peak:5.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B:5d} | OOM", flush=True); torch.cuda.empty_cache()
    if not rows:
        return {"error": "all OOM"}
    ips1 = max(r[2] for r in rows)
    ips8 = ips1 * 8 * 0.9
    price = PRICE.get(gpu, 6.25)
    print(f"\nBest single-{gpu}: {ips1:.0f} img/s -> 8x{gpu} ~{ips8:.0f} img/s @ ${price*8:.0f}/hr", flush=True)
    print(f"\n=== cost estimate (8x{gpu}, {res}px DC-AE DiT) ===", flush=True)
    print(f"{'sample-views':>13} | {'hours':>7} | {'cost $':>8}", flush=True)
    for views in (100e6, 200e6, 300e6):
        hr = views / ips8 / 3600
        print(f"{views/1e6:>10.0f} M | {hr:>7.2f} | {hr*price*8:>8.0f}", flush=True)
    print("(+ FLUX/DC-AE latent caching ~$20, distillation ~$100; no VAE training)", flush=True)
    return {"gpu": gpu, "res": res, "params_M": n / 1e6, "tokens": model.n_tokens, "best_img_s": ips1}


@app.function(image=img, gpu="B200", timeout=25 * 60)
def dit_b200(batches: str = "256,512,1024,2048", steps: int = 20, res: int = 512):
    return _dit_body("B200", batches, steps, res)


@app.function(image=img, gpu="H100", timeout=25 * 60)
def probe_h100(batches: str = "32,48,64,96", steps: int = 20, opt: int = 1, ckpt: int = 0):
    return _probe_body("H100", batches, steps, opt, ckpt)


@app.function(image=img, gpu="B200", timeout=25 * 60)
def probe_b200(batches: str = "48,64,96,128", steps: int = 20, opt: int = 1, ckpt: int = 0):
    return _probe_body("B200", batches, steps, opt, ckpt)


@app.local_entrypoint()
def main(action: str = "probe", gpu: str = "H100", batches: str = "", steps: int = 20,
         opt: int = 1, ckpt: int = 0, res: int = 512, mdl: str = ""):
    if action == "probe":
        fn = probe_b200 if gpu.upper() == "B200" else probe_h100
        kw = dict(steps=steps, opt=opt, ckpt=ckpt)
        if batches:
            kw["batches"] = batches
        print(fn.remote(**kw))
    elif action == "dit":
        kw = dict(steps=steps, res=res)
        if batches:
            kw["batches"] = batches
        print(dit_b200.remote(**kw))
    elif action == "dcae":
        r = dcae_check.remote(res=res, n=8)
        if r.get("png"):
            import base64
            with open("dcae_roundtrip.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote dcae_roundtrip.png")
        print(r)
    elif action == "cacheprobe":
        gpus = {"H100": cache_probe_h100, "L4": cache_probe_l4, "A10G": cache_probe_a10g}
        sel = [gpu.upper()] if gpu.upper() in gpus else ["H100", "L4", "A10G"]
        for g in sel:
            print(gpus[g].remote(res=res, K=(int(batches) if batches else 512), lite=opt if opt != 1 else 0))
    elif action == "cache":
        q = int(batches) if batches else 0
        print(cache_data.remote(res=res, q_flickr=q, q_cc3m=q, q_omni=q, q_mb=q) if q
              else cache_data.remote(res=res))
    elif action == "edit":                                     # image editing on held-out MagicBrush dev
        r = sample_edit.remote(n=(int(batches) if batches else 6), n_steps=(steps if steps != 20 else 50),
                               cfg=(opt if opt not in (0, 1) else 2.5), res=res,
                               mdl=(mdl or "model_cont_512"))
        if r.get("png"):
            import base64
            with open("edit_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote edit_sample.png")
        print(r)
    elif action == "gen":                                      # text->image: --batches "prompt1|prompt2|..."
        r = sample_gen.remote(prompts=batches, n_steps=(steps if steps != 20 else 50),
                              cfg=(opt if opt not in (0, 1) else 4.0), res=res, mdl=mdl)
        if r.get("png"):
            import base64
            with open("gen_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote gen_sample.png")
        print(r)
    elif action == "vol":                                      # volume utility: --batches "op,src,dst"
        a = (batches.split(",") + ["", "", ""])[:3] if batches else ["ls", "", ""]
        print(vol_op.remote(op=a[0], src=a[1], dst=a[2]))
    elif action == "cache2":                                   # 20x scaled cache: parallel shards -> merge
        n = int(batches) if batches else 4                     # n_shards (parallel H100 containers)
        sc = (opt / 100.0) if opt != 1 else 1.0                # --opt as scale*100 (e.g. 2 -> 0.02 smoke test)
        print(f"spawning {n} cache shards (scale={sc}) ...")
        handles = {i: cache_shard.spawn(res=res, shard=i, n_shards=n, scale=sc) for i in range(n)}
        for i, h in handles.items():
            try:
                print(h.get())
            except Exception as e:                             # isolate a bad shard; don't abort the merge
                print(f"shard {i} FAILED: {e}")
        print(cache_merge.remote(res=res))
    elif action == "mergecont":                                # re-run just the merge over existing cont shards
        print(cache_merge.remote(res=res, prefix="mix_cont"))
    elif action == "cacheshards":                              # re-run specific shards: --batches "8:2,3,4"
        ns, idxs = batches.split(":")
        ns = int(ns); idxs = [int(i) for i in idxs.split(",")]
        print(f"re-running shards {idxs} of {ns} ...")
        handles = {i: cache_shard.spawn(res=res, shard=i, n_shards=ns, scale=1.0) for i in idxs}
        for i, h in handles.items():
            try:
                print(h.get())
            except Exception as e:
                print(f"shard {i} FAILED: {e}")
        print(cache_merge.remote(res=res))
    elif action == "cachecont":                                # continued-pretrain cache: parallel shards -> merge
        n = int(batches) if batches else 4
        sc = (opt / 100.0) if opt != 1 else 1.0
        print(f"spawning {n} CONT cache shards (scale={sc}) ...")
        handles = {i: cache_cont_shard.spawn(res=res, shard=i, n_shards=n, scale=sc) for i in range(n)}
        for i, h in handles.items():
            try:
                print(h.get())
            except Exception as e:
                print(f"shard {i} FAILED: {e}")
        print(cache_merge.remote(res=res, prefix="mix_cont"))
    elif action == "conttrain":                                # continued pretrain: resume + edit-replay
        r = train_pilot.remote(steps=(steps if steps != 20 else 45000), res=res,
                               micro=(int(batches) if batches else 64), lr=2e-5,
                               cache="mix_cont", resume=f"/cache/model_{res}.pt",
                               replay=0.2, out=f"model_cont_{res}")
        if r.get("png"):
            import base64
            with open("cont_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote cont_sample.png")
        print(r)
    elif action == "conttrain2":                               # bigger continued pretrain: resume cont + heavy edit
        r = train_pilot.remote(steps=(steps if steps != 20 else 120000), res=res,
                               micro=(int(batches) if batches else 64), lr=2e-5,
                               cache="mix_cont", resume=(mdl or f"/cache/model_cont_{res}.pt"),
                               replay=0.25, out=f"model_cont2_{res}", save_every=15000)
        if r.get("png"):
            import base64
            with open("cont2_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote cont2_sample.png")
        print(r)
    elif action == "conttrain1024":                            # 1024px sharpness finetune (pos-embed interpolated)
        r = train_pilot.remote(steps=(steps if steps != 20 else 20000), res=1024,
                               micro=(int(batches) if batches else 16), lr=2e-5,
                               cache="mix_cont", resume=(mdl or "/cache/model_cont2_512.pt"),
                               replay=0.0, out="model_1024", save_every=5000)
        if r.get("png"):
            import base64
            with open("cont1024_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote cont1024_sample.png")
        print(r)
    elif action == "flux1024":                                 # balanced FLUX run + human boost, build on flux3
        r = train_pilot.remote(steps=(steps if steps != 20 else 100000), res=1024,
                               micro=(int(batches) if batches else 16), lr=2e-5,
                               cache="mix_cont", resume=(mdl or "/cache/model_1024flux3.pt"),
                               replay=0.0, out="model_1024flux4", save_every=5000)
        if r.get("png"):
            import base64
            with open("flux1024_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote flux1024_sample.png")
        print(r)
    elif action == "pilot":
        r = train_pilot.remote(steps=steps, res=res, micro=(int(batches) if batches else 32))
        if r.get("png"):
            import base64
            with open("pilot_sample.png", "wb") as f:
                f.write(base64.b64decode(r.pop("png")))
            print("wrote pilot_sample.png")
        print(r)
