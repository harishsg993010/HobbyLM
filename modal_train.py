"""Modal harness: download FineWeb to a Volume, then train on 1-8x H100.

  # one-time (or first-run) data download (default 10 chunks = ~1B tokens):
  python -m modal run modal_train.py --action download --chunks 10

  # train a preset (single H100 by default):
  python -m modal run modal_train.py --action train --preset 130M --steps 4000 --run-name baseline

  # train on 8x H100:
  python -m modal run modal_train.py --action train --preset 1B --gpus 8 --steps 20000

  # an ablation (override model config fields):
  python -m modal run modal_train.py --action train --preset 130M --run-name softmax --overrides "gating=softmax"
"""
import subprocess
import modal

# dataset -> (HF repo, volume dir). 10B = ~10B unique tokens; 100B = ~100B unique tokens.
DATASETS = {
    "10B": ("kjj0/fineweb10B-gpt2", "/data/fineweb10B"),
    "100B": ("kjj0/fineweb100B-gpt2", "/data/fineweb100B"),
}
DATA_DIR = "/data/fineweb10B"  # default

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "numpy", "huggingface-hub", "tqdm", "tiktoken")
    .add_local_dir(".", "/root/moe-lab")   # mounted at runtime (code iteration without rebuild)
)

app = modal.App("moe-lab", image=image)
vol = modal.Volume.from_name("fineweb10B", create_if_missing=True)


@app.function(volumes={"/data": vol}, timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")])
def download(chunks: int = 10, dataset: str = "10B"):
    import os
    from huggingface_hub import hf_hub_download
    repo, ddir = DATASETS[dataset]
    os.makedirs(ddir, exist_ok=True)
    n_done = 0

    def get(fname):
        nonlocal n_done
        if not os.path.exists(os.path.join(ddir, fname)):
            hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=ddir)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  downloaded {n_done} new files...", flush=True)
                vol.commit()  # periodic commit so a long download survives interruption

    get("fineweb_val_%06d.bin" % 0)
    for i in range(1, chunks + 1):
        get("fineweb_train_%06d.bin" % i)
    vol.commit()
    print(f"data ready [{dataset}]: {chunks} train chunks (~{chunks*100}M tokens) + val in {ddir}", flush=True)


def _train_body(preset, steps, run_name, overrides, gpus, micro, seq_len, batch_tokens, save_every=0, data_dir=DATA_DIR):
    import os
    os.chdir("/root/moe-lab")
    over = []
    if overrides:
        over = ["--set", *overrides.split(",")]
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "train.py",
        "--preset", preset, "--run_name", run_name, "--data_dir", data_dir,
        "--out_dir", "/data/runs", "--max_steps", str(steps), "--micro_batch_seqs", str(micro),
        "--seq_len", str(seq_len), "--batch_tokens", str(batch_tokens),
        "--save_every", str(save_every), *over,
    ]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    vol.commit()
    import json
    rp = f"/data/runs/{run_name}/result.json"
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    return {"run": run_name, **res}


@app.function(gpu="H100", timeout=20 * 60)
def smoke(preset: str = "130M"):
    """No-data GPU check: grouped_mm + Muon + bf16 + compile, fwd/bwd/step."""
    import os, sys, torch
    os.chdir("/root/moe-lab")
    sys.path.insert(0, "/root/moe-lab")
    from config import get_config, TrainConfig
    from model import MoETransformer, count_params
    from optim import build_optimizers
    dev = torch.device("cuda")
    cfg = get_config(preset)
    cfg.n_layers = 4  # shrink for a fast check, keep grouped backend
    model = MoETransformer(cfg).to(dev)   # fp32 master weights; bf16 via autocast
    pc = count_params(model)
    print(f"{preset} (4-layer probe): grouped backend, {pc['total']/1e6:.1f}M params", flush=True)
    muon, adamw, _ = build_optimizers(model, TrainConfig())
    cmodel = torch.compile(model)
    x = torch.randint(0, cfg.vocab_size, (4, 256), device=dev)
    y = torch.randint(0, cfg.vocab_size, (4, 256), device=dev)
    for step in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = cmodel(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        print(f"step {step}: loss={loss.item():.4f} ce={parts['ce'].item():.4f} "
              f"z={parts['z'].item():.2f} finite={torch.isfinite(loss).item()}", flush=True)
    print("GPU SMOKE OK", flush=True)


@app.function(gpu="H100", volumes={"/data": vol}, timeout=20 * 60)
def generate(run_name: str = "130M_10B", ckpt: str = "model.pt", prompt: str = "",
             max_new_tokens: int = 120, temperature: float = 0.9, top_k: int = 0):
    import os, sys
    os.chdir("/root/moe-lab")
    sys.path.insert(0, "/root/moe-lab")
    from generate import run
    ckpt_path = f"/data/runs/{run_name}/{ckpt}"
    prompts = [prompt] if prompt else None
    if prompts is None:
        prompts = [
            "The meaning of life is",
            "Once upon a time, there was a",
            "The capital of France is",
            "In 2023, scientists discovered that",
            "To make a good cup of coffee, you",
            "Artificial intelligence will",
        ]
    run(ckpt_path, prompts, max_new_tokens, temperature, top_k)


@app.function(gpu="H100", volumes={"/data": vol}, timeout=20 * 60)
def specdecode(draft_run: str = "130M_10B", target_run: str = "500M_40B", K: int = 4, prompt: str = ""):
    """Speculative decoding: draft proposes K tokens, target verifies in one pass. Reports speedup."""
    import os, sys
    os.chdir("/root/moe-lab")
    sys.path.insert(0, "/root/moe-lab")
    from spec_decode import run
    prompts = [prompt] if prompt else ["The meaning of life is",
                                        "Once upon a time, there was a",
                                        "In 2023, scientists discovered that"]
    run(f"/data/runs/{draft_run}/model.pt", f"/data/runs/{target_run}/model.pt",
        prompts, max_new_tokens=100, K=K, temperature=0.8, top_p=0.95)


@app.function(gpu="H100", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_1(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0, data_dir=DATA_DIR):
    return _train_body(preset, steps, run_name, overrides, 1, micro, seq_len, batch_tokens, save_every, data_dir)


@app.function(gpu="H100:8", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_8(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0, data_dir=DATA_DIR):
    return _train_body(preset, steps, run_name, overrides, 8, micro, seq_len, batch_tokens, save_every, data_dir)


# ---- focused ablation suite (each changes ONE thing vs the 130M baseline) ----
ABLATIONS: dict[str, str] = {
    "baseline": "",                                   # aux_free, sigmoid, no-shared, 32 exp / top4
    "softmax": "gating=softmax",                      # gating fn fork
    "aux_loss": "balancing=aux_loss,aux_loss_coef=0.01",  # classic balance loss vs aux-free bias
    "shared1": "n_shared=1",                          # add 1 always-on shared expert
    "no_qknorm": "qk_norm=false",                     # QK-norm on/off
    "no_renorm": "norm_topk_prob=false",              # renormalize top-k gate weights
    "topk8": "top_k=8",                               # more active experts
    "experts16": "n_experts=16",                      # fewer total experts
    "no_zloss": "z_loss_coef=0.0",                    # router z-loss on/off
    "scale_emb": "scale_embeddings=true",             # Gemma sqrt(d) embedding scale
}


@app.function(volumes={"/data": vol}, timeout=60)
def results():
    import glob, json, os
    rows = []
    for f in glob.glob("/data/runs/*/result.json"):
        name = os.path.basename(os.path.dirname(f))
        try:
            rows.append((name, json.load(open(f)).get("final_val_loss")))
        except Exception:
            pass
    rows.sort(key=lambda r: (r[1] is None, r[1]))
    print(f"{'run':>14} | {'final_val_loss':>14}")
    print("-" * 33)
    for n, v in rows:
        print(f"{n:>14} | {v:>14.4f}" if v is not None else f"{n:>14} | {'(running)':>14}")


@app.local_entrypoint()
def main(action: str = "train", preset: str = "130M", steps: int = 4000,
         run_name: str = "baseline", overrides: str = "", gpus: int = 1,
         chunks: int = 10, micro: int = 16, seq_len: int = 1024,
         batch_tokens: int = 262144, save_every: int = 0, data: str = "10B"):
    data_dir = DATASETS[data][1]
    if action == "download":
        download.remote(chunks, data)
    elif action == "smoke":
        smoke.remote(preset)
    elif action == "train":
        fn = train_8 if gpus == 8 else train_1
        fn.remote(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every, data_dir)
    elif action == "ablate":
        # run all ablations in parallel via starmap (keeps the app alive until all finish)
        arg_tuples = [(preset, steps, f"ab_{name}", ov, micro, seq_len, batch_tokens)
                      for name, ov in ABLATIONS.items()]
        print(f"launching {len(arg_tuples)} ablations at {preset}, {steps} steps each (parallel H100s)...",
              flush=True)
        done = []
        for res in train_1.starmap(arg_tuples):
            done.append(res)
            print(f"  done: {res.get('run'):>16}  val_loss={res.get('final_val_loss')}", flush=True)
        done.sort(key=lambda r: (r.get("final_val_loss") is None, r.get("final_val_loss")))
        print("\n=== ABLATION LEADERBOARD (final val loss) ===", flush=True)
        for r in done:
            print(f"{r.get('run'):>16} | {r.get('final_val_loss')}", flush=True)
    elif action == "results":
        results.remote()
    elif action == "generate":
        generate.remote(run_name, "model.pt", overrides, 120, 0.9, 0)
    elif action == "specdecode":
        specdecode.remote("130M_10B", "500M_40B", 4, overrides)
    else:
        raise SystemExit(f"unknown action {action!r} (use download|smoke|train|ablate|results|generate)")
