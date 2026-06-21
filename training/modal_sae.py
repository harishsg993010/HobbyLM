"""Train + analyze a Top-k Sparse Autoencoder on HobbyLM residual-stream activations (mech interp).

Online training: stream FineWeb tokens through the FROZEN base model up to `--layer`, capture the
residual stream, and train the SAE on those activations (no 150 GB activation cache needed).

  python -m modal run training/modal_sae.py --action train  --layer 8 --tokens 50000000
  python -m modal run training/modal_sae.py --action analyze --sae 500M_40B_L8_sae --feats 24
"""
import modal

app = modal.App("hobbylm-sae")

img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch", "numpy", "tiktoken", "safetensors", "huggingface_hub>=0.25.0")
       .add_local_dir(".", "/root/moe-lab", ignore=["hobby-chat/**", "hobby-rs/**", "hobby-rs-cli/**",
                       "**/target/**", "*.gguf", "*.pt", "*.safetensors", "checkpoints/**",
                       "image_weights/**", "needle/**", "Megatron-Bridge/**", "space/**", "*.png"]))

vol = modal.Volume.from_name("fineweb10B")
HF_SECRET = modal.Secret.from_name("huggingface")
DATA_GLOB = "/data/fineweb10B/fineweb_train_*.bin"
# default = the model HobbyLM-Base ships (so an uploaded SAE matches the public weights)
DEFAULT_MODEL_RUN = "500M_ctx8k"


@app.function(image=img, gpu="H100", volumes={"/data": vol}, timeout=6 * 60 * 60)
def train(layer: int = 8, tokens: int = 50_000_000, d_sae: int = 12288, k: int = 32,
          lr: float = 4e-4, micro: int = 4096, buf_seqs: int = 256, seq: int = 1024,
          save_name: str = "", model_run: str = DEFAULT_MODEL_RUN):
    import os, sys, time, math, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from hobbylm.generate import load_model
    from hobbylm.data import data_generator
    from hobbylm.sae import TopKSAE, SAEConfig, fraction_variance_explained
    dev = torch.device("cuda")
    torch.manual_seed(0); torch.set_float32_matmul_precision("high")

    MODEL = f"/data/runs/{model_run}/model.pt"
    model, cfg, vloss, _ = load_model(MODEL, dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = cfg.d_model
    save_name = save_name or f"{model_run}_L{layer}_sae"
    print(f"model d={d} L={cfg.n_layers} val={vloss:.3f} | capture residual after block {layer} | "
          f"d_sae={d_sae} k={k}", flush=True)

    sae = TopKSAE(SAEConfig(d_in=d, d_sae=d_sae, k=k)).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))

    gen = data_generator(DATA_GLOB, B=buf_seqs, S=seq, device=dev, to_device=True)

    @torch.no_grad()
    def acts_for(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(ids, capture_layer=layer)            # (B, S, d)
        return h.reshape(-1, d).float()

    scale = None
    inited = False
    seen = 0
    step = 0
    t0 = time.time()
    while seen < tokens:
        x_ids, _ = next(gen)                               # (buf_seqs, seq)
        buf = acts_for(x_ids)                              # (buf_seqs*seq, d)
        if scale is None:                                  # normalize acts to RMS-norm ~ sqrt(d)
            scale = (d ** 0.5) / buf.norm(dim=-1).mean().clamp_min(1e-6)
            scale = float(scale)
            print(f"activation scale = {scale:.4f}", flush=True)
        buf = buf * scale
        if not inited:
            sae.set_decoder_to_geometric_mean(buf); inited = True
        perm = torch.randperm(buf.shape[0], device=dev)
        for i in range(0, buf.shape[0] - micro + 1, micro):
            xb = buf[perm[i:i + micro]]
            _, _, m = sae(xb)
            opt.zero_grad(set_to_none=True)
            m["loss"].backward()
            opt.step()
            sae.normalize_decoder()
            step += 1
            if step % 100 == 0:
                with torch.no_grad():
                    xh, z, _ = sae(xb)
                    fve = fraction_variance_explained(xb, xh)
                    l0 = float((z > 0).float().sum(-1).mean())
                    dead = int((sae.last_fired > sae.cfg.dead_after).sum())
                print(f"step {step:6d} | tok {seen/1e6:5.1f}M | recon {m['recon']:.4f} | "
                      f"var_expl {fve*100:5.1f}% | L0 {l0:.0f} | dead {dead}/{d_sae} | "
                      f"{(time.time()-t0)/step*1000:.0f}ms/step", flush=True)
        seen += buf.shape[0]

    out = f"/data/saes/{save_name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"sae": sae.state_dict(), "cfg": vars(sae.cfg), "layer": layer,
                "scale": scale, "model": MODEL, "tokens": seen}, f"{out}/sae.pt")
    vol.commit()
    with torch.no_grad():
        xh, z, _ = sae(buf[:micro])
        fve = fraction_variance_explained(buf[:micro], xh)
    print(f"DONE -> {out}/sae.pt | {seen/1e6:.0f}M tokens | var_expl {fve*100:.1f}% | "
          f"dead {int((sae.last_fired > sae.cfg.dead_after).sum())}/{d_sae}", flush=True)
    return {"sae": f"{out}/sae.pt", "tokens": seen}


@app.function(image=img, gpu="H100", volumes={"/data": vol}, timeout=2 * 60 * 60)
def analyze(sae_name: str = "500M_40B_L8_sae", feats: int = 24, tokens: int = 4_000_000,
            topk: int = 6, ctx: int = 8, seq: int = 1024, buf_seqs: int = 64):
    """Find the max-activating token contexts for a sample of SAE features (the interpretability view)."""
    import os, sys, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from hobbylm.generate import load_model
    from hobbylm.data import data_generator
    from hobbylm.sae import TopKSAE, SAEConfig
    dev = torch.device("cuda")
    enc = tiktoken.get_encoding("gpt2")
    ck = torch.load(f"/data/saes/{sae_name}/sae.pt", map_location=dev, weights_only=False)
    layer, scale = ck["layer"], ck["scale"]
    sae = TopKSAE(SAEConfig(**ck["cfg"])).to(dev)
    sae.load_state_dict(ck["sae"]); sae.eval()
    model, cfg, _, _ = load_model(ck["model"], dev); model.eval()
    d = cfg.d_model
    m = sae.cfg.d_sae
    # a representative spread of feature indices to inspect
    feat_ids = torch.linspace(0, m - 1, feats).long().tolist()
    fset = torch.tensor(feat_ids, device=dev)

    best_val = torch.full((feats, topk), -1.0, device=dev)
    best_tok = [[None] * topk for _ in range(feats)]       # (token-window ids) per slot
    gen = data_generator(DATA_GLOB, B=buf_seqs, S=seq, device=dev, to_device=True)
    seen = 0
    while seen < tokens:
        x_ids, _ = next(gen)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(x_ids, capture_layer=layer).float() * scale
        z = sae.encode(h.reshape(-1, d))                   # (B*S, m)
        zf = z[:, fset]                                    # (B*S, feats)
        ids_flat = x_ids.reshape(-1)
        for fi in range(feats):
            col = zf[:, fi]
            v, p = col.topk(min(topk, col.numel()))
            for slot in range(len(v)):
                if float(v[slot]) > float(best_val[fi].min()):
                    j = int(best_val[fi].argmin())
                    if float(v[slot]) > float(best_val[fi, j]):
                        pos = int(p[slot]); lo = max(0, pos - ctx); hi = min(ids_flat.numel(), pos + 2)
                        best_val[fi, j] = float(v[slot])
                        best_tok[fi][j] = (ids_flat[lo:pos].tolist(), int(ids_flat[pos]), ids_flat[pos + 1:hi].tolist())
        seen += x_ids.numel()

    print(f"\n=== {sae_name} | layer {layer} | {seen/1e6:.1f}M tokens | top-activating contexts ===\n", flush=True)
    for fi, f in enumerate(feat_ids):
        order = best_val[fi].argsort(descending=True)
        print(f"--- feature #{f} ---", flush=True)
        for j in order.tolist():
            t = best_tok[fi][j]
            if t is None or best_val[fi, j] < 0:
                continue
            pre = enc.decode(t[0]).replace("\n", " ")
            cur = enc.decode([t[1]])
            post = enc.decode(t[2]).replace("\n", " ")
            print(f"   {best_val[fi,j]:6.2f}  …{pre}⟦{cur}⟧{post}…", flush=True)
        print(flush=True)
    return {"sae": sae_name, "feats": feat_ids}


@app.function(image=img, gpu="H100", volumes={"/data": vol}, timeout=2 * 60 * 60)
def labels(sae_name: str = "500M_ctx8k_L8_sae", tokens: int = 8_000_000, top: int = 12,
           seq: int = 1024, buf_seqs: int = 16):
    """Auto-label every SAE feature by its top max-activating tokens, and export the SAE as safetensors.
    Writes labels.json + sae.safetensors next to sae.pt — what the Space loads."""
    import os, sys, json, torch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from collections import Counter
    from safetensors.torch import save_file
    from hobbylm.generate import load_model
    from hobbylm.data import data_generator
    from hobbylm.sae import TopKSAE, SAEConfig
    dev = torch.device("cuda")
    enc = tiktoken.get_encoding("gpt2")
    sdir = f"/data/saes/{sae_name}"
    ck = torch.load(f"{sdir}/sae.pt", map_location=dev, weights_only=False)
    layer, scale = ck["layer"], ck["scale"]
    sae = TopKSAE(SAEConfig(**ck["cfg"])).to(dev); sae.load_state_dict(ck["sae"]); sae.eval()
    model, cfg, _, _ = load_model(ck["model"], dev); model.eval()
    d, m = cfg.d_model, sae.cfg.d_sae

    # per-feature: keep the top `top` (value, token-id) so we can name it by its most frequent token
    best_val = torch.full((m, top), -1.0, device=dev)
    best_tid = torch.zeros((m, top), dtype=torch.long, device=dev)
    gen = data_generator(DATA_GLOB, B=buf_seqs, S=seq, device=dev, to_device=True)
    seen = 0
    while seen < tokens:
        x_ids, _ = next(gen)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model(x_ids, capture_layer=layer).float() * scale
            z = sae.encode(h.reshape(-1, d))              # (N, m) — MUST be no_grad (else graph leaks)
        ids_flat = x_ids.reshape(-1)
        # per-feature top activations on this buffer, merged into the running global top list.
        # top-k along the TOKEN axis (dim 0) — avoids materializing the (m, N) transpose.
        kk = min(top, z.shape[0])
        bv, bp = z.topk(kk, dim=0)                          # (kk, m) values + token-row indices
        bv = bv.t().contiguous()                           # (m, kk)
        btoks = ids_flat[bp].t().contiguous()              # (m, kk)
        cand_v = torch.cat([best_val, bv], dim=1)
        cand_t = torch.cat([best_tid, btoks], dim=1)
        order = cand_v.topk(top, dim=1).indices
        best_val = torch.gather(cand_v, 1, order)
        best_tid = torch.gather(cand_t, 1, order)
        seen += x_ids.numel()
        del z, h, bv, bp, btoks, cand_v, cand_t

    out = {}
    btid = best_tid.cpu().tolist(); bval = best_val.cpu().tolist()
    for f in range(m):
        toks = [enc.decode([t]) for t, v in zip(btid[f], bval[f]) if v > 0]
        if not toks:
            out[f] = {"label": "(dead)", "max_act": 0.0}; continue
        common = Counter([t.strip() or t for t in toks]).most_common(1)[0][0]
        out[f] = {"label": common[:24], "top_tokens": toks[:6], "max_act": round(bval[f][0], 2)}
    json.dump(out, open(f"{sdir}/labels.json", "w"))
    save_file({k2: v.cpu().contiguous() for k2, v in sae.state_dict().items()}, f"{sdir}/sae.safetensors")
    json.dump({"cfg": ck["cfg"], "layer": layer, "scale": scale, "model_run": ck["model"]},
              open(f"{sdir}/meta.json", "w"), indent=1)
    vol.commit()
    named = sum(1 for v in out.values() if v["label"] != "(dead)")
    print(f"labeled {named}/{m} features -> {sdir}/labels.json + sae.safetensors", flush=True)
    return {"named": named, "total": m}


@app.function(image=img, volumes={"/data": vol}, secrets=[HF_SECRET], timeout=20 * 60)
def upload(sae_name: str = "500M_ctx8k_L8_sae", repo: str = "rootxhacker/HobbyLM-SAE"):
    """Push sae.safetensors + labels.json + meta.json + a README to a (public) HF repo."""
    from huggingface_hub import HfApi
    import json
    sdir = f"/data/saes/{sae_name}"
    meta = json.load(open(f"{sdir}/meta.json"))
    labels_n = sum(1 for v in json.load(open(f"{sdir}/labels.json")).values() if v["label"] != "(dead)")
    readme = f"""---
license: apache-2.0
tags: [hobbylm, sparse-autoencoder, interpretability, sae]
---

# HobbyLM-SAE

A **top-k Sparse Autoencoder** for mechanistic interpretability of [HobbyLM-Base](https://huggingface.co/rootxhacker/HobbyLM-Base).
It decomposes the residual stream after **layer {meta['layer']}** into a sparse, overcomplete dictionary of
**{meta['cfg']['d_sae']} features** ({meta['cfg']['k']} active per token), most of them human-interpretable
({labels_n} auto-labeled by their top-activating tokens).

## Files
- `sae.safetensors` — the SAE weights (`W_enc`, `W_dec`, `b_enc`, `b_dec`).
- `labels.json` — per-feature auto-derived label + example top-activating tokens.
- `meta.json` — layer, activation scale, base-model run, and SAE config.

Reconstructs ~97% of the activation variance at L0={meta['cfg']['k']}. Reference code + training harness:
<https://github.com/harishsg993010/HobbyLM> (`hobbylm/sae.py`, `training/modal_sae.py`). Apache-2.0.
"""
    open(f"{sdir}/README.md", "w").write(readme)
    api = HfApi()
    api.create_repo(repo, private=False, exist_ok=True, repo_type="model")
    for f in ["sae.safetensors", "labels.json", "meta.json", "README.md"]:
        api.upload_file(path_or_fileobj=f"{sdir}/{f}", path_in_repo=f, repo_id=repo, repo_type="model")
    print(f"pushed SAE -> https://huggingface.co/{repo}", flush=True)
    return repo


@app.local_entrypoint()
def main(action: str = "train", layer: int = 8, tokens: int = 50_000_000, d_sae: int = 12288,
         k: int = 32, sae: str = "500M_ctx8k_L8_sae", feats: int = 24, model_run: str = DEFAULT_MODEL_RUN):
    if action == "train":
        print(train.remote(layer=layer, tokens=tokens, d_sae=d_sae, k=k, model_run=model_run))
    elif action == "analyze":
        print(analyze.remote(sae_name=sae, feats=feats))
    elif action == "labels":
        print(labels.remote(sae_name=sae))
    elif action == "upload":
        print(upload.remote(sae_name=sae))
    else:
        raise SystemExit(f"unknown action {action!r} (train|analyze|labels|upload)")
