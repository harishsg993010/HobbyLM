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

# separate image for lm-evaluation-harness (heavy deps: transformers/datasets) so it doesn't
# perturb the training image. torch pinned first so lm-eval doesn't pull a different build.
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "numpy", "tiktoken", "lm-eval==0.4.9.1",
                 "transformers>=4.50,<5", "datasets", "huggingface-hub", "accelerate", "sentencepiece")
    # transformers <5 (lm-eval needs AutoModelForVision2Seq) but >=4.50 (for Gemma-3 / Qwen2.5 archs)
    # route downloads via std CDN (xet endpoint rate-limits under parallelism); cache datasets/models
    # on a shared volume so parallel eval containers don't each re-download (avoids HF 429s).
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(".", "/root/moe-lab")
)

app = modal.App("moe-lab", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  # HF datasets/models cache
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


# ---- throughput-optimization presets: (model-config --set overrides, orthogonalizer) ----
# SHIPPING: all_safe (fused_ce + polar). fused_ce is the win: +6% step time, -21% peak memory,
# enables ~2x larger batch, numerics-identical.
# EXPERIMENTAL (not recommended): fp8 / all_max. The fp8 head gave NO speedup at 1B, OOM'd at large
# batch, added 51M params, and — as the 130M ablation showed — does not train (zero gradient flows
# back through mm_t_backward, loss frozen at init). Kept here only for reference; needs a backward fix.
OPT_PRESETS: dict[str, tuple[list, str]] = {
    "baseline": ([], "ns5"),
    "fused_ce": (["fused_ce=true"], "ns5"),
    "polar":    ([], "polar"),
    "all_safe": (["fused_ce=true"], "polar"),
    "fp8":      (["fp8_head=true"], "ns5"),       # experimental — broken, see note above
    "all_max":  (["fp8_head=true"], "polar"),     # experimental — broken
}


def _train_body(preset, steps, run_name, overrides, gpus, micro, seq_len, batch_tokens,
                save_every=0, data_dir=DATA_DIR, opts="baseline", init_from="", lr_mult=1.0):
    import os
    os.chdir("/root/moe-lab")
    opt_sets, orthog = OPT_PRESETS.get(opts, ([], "ns5"))
    user_sets = overrides.split(",") if overrides else []
    all_sets = [*opt_sets, *user_sets]
    over = ["--set", *all_sets] if all_sets else []
    extra = []
    if init_from:
        extra += ["--init_from", init_from]
    if lr_mult != 1.0:
        extra += ["--lr_mult", str(lr_mult)]
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "train.py",
        "--preset", preset, "--run_name", run_name, "--data_dir", data_dir,
        "--out_dir", "/data/runs", "--max_steps", str(steps), "--micro_batch_seqs", str(micro),
        "--seq_len", str(seq_len), "--batch_tokens", str(batch_tokens),
        "--orthogonalizer", orthog, "--save_every", str(save_every), *extra, *over,
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
def gpu_bench(diff_run: str = "500M_diff_20b", ar_run: str = "500M_40B",
              gen_len: int = 128, block: int = 32, steps: int = 32):
    """AR vs diffusion tok/s on GPU. Measures: (1) per-forward latency vs seq length (does the GPU
    make a batched canvas-forward ~free? = the crux of whether diffusion can win), (2) end-to-end
    tok/s for autoregressive decode vs iterative-denoising decode, (3) forwards-per-token."""
    import os, sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from generate import load_model
    from diffusion import generate as dgen
    dev = torch.device("cuda")
    enc = tiktoken.get_encoding("gpt2")
    diff_model, dcfg, _, _ = load_model(f"/data/runs/{diff_run}/model.pt", dev)
    ar_model, _, _, _ = load_model(f"/data/runs/{ar_run}/model.pt", dev)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def time_forward(model, L, iters=30):
        x = torch.randint(0, 50000, (1, L), device=dev)
        with amp():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
        return (time.time() - t) / iters * 1000  # ms/forward

    print("=== per-forward latency vs seq length (diffusion model, bidirectional, bf16) ===", flush=True)
    lat = {}
    for L in [1, 16, 32, 64, 128, 256]:
        lat[L] = time_forward(diff_model, L)
        print(f"  L={L:4d}: {lat[L]:6.2f} ms/forward   ({lat[L]/lat[1]:.2f}x of L=1)", flush=True)

    ids = torch.tensor([enc.encode_ordinary("The capital of France is")], device=dev)

    # AR end-to-end, naive (no KV cache — the training model has none), as a lower bound on AR
    with torch.no_grad(), amp():
        torch.cuda.synchronize(); t = time.time()
        x = ids.clone()
        for _ in range(gen_len):
            logits, _ = ar_model(x[:, -1024:])
            x = torch.cat([x, logits[:, -1].argmax(-1, keepdim=True)], 1)
        torch.cuda.synchronize()
    ar_naive = gen_len / (time.time() - t)

    # diffusion end-to-end, counting forwards via a pre-hook
    count = [0]
    h = diff_model.register_forward_pre_hook(lambda m, i: count.__setitem__(0, count[0] + 1))
    with torch.no_grad(), amp():
        torch.cuda.synchronize(); t = time.time()
        out = dgen(diff_model, ids, gen_len=gen_len, block=block, steps=steps,
                   mask_id=dcfg.mask_token_id, temperature=0.0, rep_penalty=1.4, remask_steps=0)
        torch.cuda.synchronize()
    diff_dt = time.time() - t
    h.remove()
    ntok = out.shape[1]

    print(f"\n=== end-to-end ({gen_len} tokens) ===", flush=True)
    print(f"AR naive (no KV cache, LOWER bound):     {ar_naive:6.1f} tok/s", flush=True)
    print(f"AR cached estimate (1 fwd/tok @ L=1):    {1000 / lat[1]:6.1f} tok/s", flush=True)
    print(f"Diffusion (steps={steps} block={block}): {ntok / diff_dt:6.1f} tok/s "
          f"| {count[0]} forwards = {count[0] / max(ntok,1):.2f} fwd/tok (AR cached = 1.0)", flush=True)
    print(f"\nReading: AR-cached does ~1 fwd/token @ ~{lat[1]:.1f}ms; diffusion does "
          f"{count[0]/max(ntok,1):.2f} fwd/token but each over the whole canvas (~{lat.get(128,0):.1f}ms @ L=128). "
          f"Diffusion wins iff (fwd/tok x canvas_ms) < (1 x {lat[1]:.1f}ms).", flush=True)
    return {"ar_naive": ar_naive, "ar_cached_est": 1000 / lat[1],
            "diff_toks_per_s": ntok / diff_dt, "diff_fwd_per_tok": count[0] / max(ntok, 1), "lat": lat}


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
def diffgen(run_name: str = "500M_diff_5b", ckpt: str = "ckpt_1000.pt", prompt: str = "",
            gen_len: int = 64, block: int = 32, steps: int = 64, temperature: float = 0.0,
            rep_penalty: float = 1.0, remask_steps: int = 0, remask_frac: float = 0.3,
            sweep: int = 0):
    """Iterative-denoising sample from a pure-diffusion (LLaDA) checkpoint. The saved config
    carries diffusion=True, so load_model rebuilds it bidirectional automatically."""
    import os, sys, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from generate import load_model
    from diffusion import generate as dgen
    dev = torch.device("cuda")
    ckpt_path = f"/data/runs/{run_name}/{ckpt}"
    model, cfg, val_loss, step = load_model(ckpt_path, dev)
    assert cfg.diffusion, f"{ckpt_path} is not a diffusion model (cfg.diffusion=False)"
    enc = tiktoken.get_encoding("gpt2")
    prompts = [prompt] if prompt else [
        "The capital of France is",
        "Once upon a time, there was a",
        "The meaning of life is",
        "Water boils at a temperature of",
    ]
    print(f"loaded {ckpt_path} | step={step} val_loss={val_loss} | diffusion={cfg.diffusion} "
          f"mask_id={cfg.mask_token_id}", flush=True)
    # decode-config sweep: load model once, try several denoise settings on the same prompts.
    if sweep:
        configs = [
            ("baseline      ", dict(block=32, steps=64,  temperature=0.7, rep_penalty=1.2, remask_steps=2)),
            ("lowtemp       ", dict(block=32, steps=128, temperature=0.3, rep_penalty=1.3, remask_steps=2)),
            ("greedy-careful", dict(block=16, steps=128, temperature=0.0, rep_penalty=1.3, remask_steps=2)),
            ("ar-ish-block8 ", dict(block=8,  steps=128, temperature=0.2, rep_penalty=1.4, remask_steps=3)),
            ("greedy-strong ", dict(block=32, steps=96,  temperature=0.0, rep_penalty=1.5, remask_steps=3)),
        ]
        prompts = prompts[:3]
    else:
        configs = [("single        ", dict(block=block, steps=steps, temperature=temperature,
                                            rep_penalty=rep_penalty, remask_steps=remask_steps))]
    for label, c in configs:
        print("\n" + "#" * 78, flush=True)
        print(f"# CONFIG [{label}] gen_len={gen_len} {c}", flush=True)
        print("#" * 78, flush=True)
        for p in prompts:
            ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dgen(model, ids, gen_len=gen_len, mask_id=cfg.mask_token_id, eos_id=50256,
                           remask_frac=remask_frac, **c)
            cont = enc.decode([t for t in out[0].tolist() if t < 50257])
            print(f"  PROMPT: {p!r}")
            print(f"  CONT:   {cont!r}\n", flush=True)


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
def train_1(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0,
            data_dir=DATA_DIR, opts="baseline", init_from="", lr_mult=1.0):
    return _train_body(preset, steps, run_name, overrides, 1, micro, seq_len, batch_tokens,
                       save_every, data_dir, opts, init_from, lr_mult)


@app.function(gpu="H100:8", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_8(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0,
            data_dir=DATA_DIR, opts="baseline", init_from="", lr_mult=1.0):
    return _train_body(preset, steps, run_name, overrides, 8, micro, seq_len, batch_tokens,
                       save_every, data_dir, opts, init_from, lr_mult)


@app.function(gpu="B200:8", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_8b200(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0,
                data_dir=DATA_DIR, opts="baseline", init_from="", lr_mult=1.0):
    return _train_body(preset, steps, run_name, overrides, 8, micro, seq_len, batch_tokens,
                       save_every, data_dir, opts, init_from, lr_mult)


# default benchmark suite for small models (all loglikelihood / multiple-choice, GPT-2/Pythia-style)
EVAL_TASKS = "lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,sciq,boolq"
# MicroLlama / TinyLlama 7-task comparison set (acc_norm except winogrande/boolq -> acc)
EVAL_TASKS_7 = "hellaswag,openbookqa,winogrande,arc_challenge,arc_easy,boolq,piqa"


def _pick_metric(task: str, metrics: dict):
    """Return (base_metric_name, value) matching the MicroLlama/TinyLlama convention:
    acc_norm where defined, else acc."""
    for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if key in metrics:
            return key.split(",")[0], metrics[key]
    k = next((k for k in metrics if not k.endswith("_stderr") and k != "alias"), None)
    return (k.split(",")[0], metrics[k]) if k else (None, None)


def _summarize(res: dict, requested: list) -> dict:
    """task -> score*100, only for the REQUESTED top-level tasks. Grouped tasks (e.g. mmlu has 57
    subtasks) report just the group aggregate, not every subtask."""
    results = res.get("results", {})
    groups = res.get("groups", {})
    out = {}
    for t in requested:
        m = results.get(t) or groups.get(t)
        if not m:
            continue
        base, val = _pick_metric(t, m)
        if base is not None and val is not None:
            out[t] = round(100 * float(val), 2)
    return out


# reference open models across sizes for an apples-to-apples comparison (run through OUR harness).
# (name -> (hf_id, approx total params, pretrain tokens) for the annotated table)
REF_MODELS = {
    # classic baselines
    "pythia-160m":       ("EleutherAI/pythia-160m", "160M", "300B"),
    "pythia-410m":       ("EleutherAI/pythia-410m", "410M", "300B"),
    "pythia-1.4b":       ("EleutherAI/pythia-1.4b", "1.4B", "300B"),
    "gpt2-124m":         ("openai-community/gpt2", "124M", "~10B"),
    "opt-350m":          ("facebook/opt-350m", "350M", "180B"),
    "opt-1.3b":          ("facebook/opt-1.3b", "1.3B", "180B"),
    # the user's reference rows
    "MicroLlama-300M":   ("keeeeenw/MicroLlama", "300M", "50B"),
    "TinyLlama-1.1B-3T": ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "1.1B", "3T"),
    # latest small models (2025)
    "SmolLM2-135M":      ("HuggingFaceTB/SmolLM2-135M", "135M", "2T"),
    "SmolLM2-360M":      ("HuggingFaceTB/SmolLM2-360M", "360M", "4T"),
    "SmolLM2-1.7B":      ("HuggingFaceTB/SmolLM2-1.7B", "1.7B", "11T"),
    "Qwen3-0.6B":        ("Qwen/Qwen3-0.6B-Base", "600M", "36T"),
    "Qwen3-1.7B":        ("Qwen/Qwen3-1.7B-Base", "1.7B", "36T"),
    "gemma-3-270m":      ("google/gemma-3-270m", "270M", "6T"),       # gated (needs HF license accepted)
    "gemma-3-1b":        ("google/gemma-3-1b-pt", "1.0B", "2T"),      # gated
}

# our own measured 7-task numbers (MicroLlama convention; --action lmeval, fixed _encode_pair, 2026-05-31)
OUR_RESULTS = {
    "OURS-130M-MoE": {"params": "140M/62M act", "tokens": "10B",
                      "scores": {"hellaswag": 32.54, "openbookqa": 28.20, "winogrande": 52.17,
                                 "arc_challenge": 23.46, "arc_easy": 37.71, "boolq": 61.31, "piqa": 65.40},
                      "avg": 42.97},
    "OURS-500M-MoE": {"params": "500M/169M act", "tokens": "40B",
                      "scores": {"hellaswag": 41.56, "openbookqa": 29.80, "winogrande": 51.30,
                                 "arc_challenge": 22.44, "arc_easy": 42.76, "boolq": 50.98, "piqa": 69.53},
                      "avg": 44.05},
}


@app.function(image=eval_image, gpu="H100", timeout=3 * 60 * 60, volumes={"/cache/hf": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def hf_eval(name: str, model_id: str, tasks: str = EVAL_TASKS_7,
            limit: int = 0, batch_size: str = "auto", num_fewshot: int = 0):
    """Eval a HuggingFace model through the SAME lm-eval harness/protocol as our models.
    batch_size='auto' so big-vocab models (Gemma-3 262k, Qwen3 151k) don't OOM in log_softmax."""
    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from lm_eval.evaluator import simple_evaluate
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    print(f"[hf_eval] {name} ({model_id})", flush=True)
    requested = [t for t in tasks.split(",") if t]
    res = simple_evaluate(model="hf",
                          model_args=f"pretrained={model_id},dtype={dtype},trust_remote_code=True",
                          tasks=requested, limit=(limit or None), num_fewshot=num_fewshot, batch_size=batch_size)
    summary = _summarize(res, requested)
    avg = round(sum(summary.values()) / len(summary), 2) if summary else None
    print(f"[hf_eval] {name}: {summary} avg={avg}", flush=True)
    return {"name": name, "model_id": model_id, "scores": summary, "avg": avg}


@app.function(image=eval_image, gpu="H100", volumes={"/data": vol, "/cache/hf": hf_cache},
              timeout=4 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def lm_eval_run(run_name: str, ckpt: str = "model.pt", tasks: str = EVAL_TASKS,
                limit: int = 0, batch_size: int = 32, num_fewshot: int = 0):
    """Run the EleutherAI lm-evaluation-harness on a trained checkpoint."""
    import os, sys, json
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import torch
    from lm_eval.evaluator import simple_evaluate
    from generate import load_model
    from eval_harness import MoELMWrapper

    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    ckpt_path = f"/data/runs/{run_name}/{ckpt}"
    model, cfg, val_loss, step = load_model(ckpt_path, dev)
    print(f"loaded {ckpt_path} | step={step} val_loss={val_loss} | "
          f"d{cfg.d_model} L{cfg.n_layers} {cfg.n_experts}exp/top{cfg.top_k}", flush=True)

    lm = MoELMWrapper(model, dev, max_length=1024, batch_size=batch_size)
    task_list = [t for t in tasks.split(",") if t]
    res = simple_evaluate(model=lm, tasks=task_list,
                          limit=(limit or None), num_fewshot=num_fewshot, batch_size=batch_size)

    # ---- print a clean table + persist results.json next to the checkpoint ----
    # only report the requested top-level tasks (grouped tasks like mmlu would otherwise dump 57 subtasks)
    scores = _summarize(res, task_list)              # task -> value*100
    print(f"\n=== lm-eval [{run_name}] ({num_fewshot}-shot, limit={limit or 'full'}) ===", flush=True)
    print(f"{'task':>16} | {'value':>7}")
    print("-" * 28)
    summary = {}
    for task in task_list:
        if task not in scores:
            continue
        m = res["results"].get(task) or res.get("groups", {}).get(task, {})
        base, _ = _pick_metric(task, m)
        summary[task] = {base: scores[task] / 100.0}
        print(f"{task:>16} | {scores[task]:>7.2f}", flush=True)
    if scores:
        print("-" * 28)
        print(f"{'AVERAGE':>16} | {sum(scores.values())/len(scores):>7.2f}", flush=True)

    out = {"run": run_name, "step": step, "val_loss": val_loss, "num_fewshot": num_fewshot,
           "limit": limit, "results": res["results"]}
    try:
        with open(f"/data/runs/{run_name}/lm_eval.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        vol.commit()
    except Exception as e:
        print(f"(could not save lm_eval.json: {e})", flush=True)
    return summary


@app.function(image=eval_image, timeout=2 * 60 * 60, volumes={"/cache/hf": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def prep_data(tasks: str = "mmlu"):
    """One-time: download the eval datasets to the shared HF cache volume from a SINGLE container
    (sequential, no parallel 429 storm). Subsequent parallel evals hit the cache instead of HF."""
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager()
    task_list = [t for t in tasks.split(",") if t]
    print(f"caching datasets for {task_list} ...", flush=True)
    td = get_task_dict(task_list, tm)   # building tasks triggers dataset download to HF_HOME=/cache/hf

    def _touch(d):
        for k, v in (d.items() if isinstance(d, dict) else []):
            if isinstance(v, dict):
                _touch(v)
            else:
                try:
                    (v.test_docs() or v.validation_docs())
                except Exception:
                    pass
    _touch(td)
    hf_cache.commit()
    print(f"cached {len(task_list)} task group(s) to /cache/hf", flush=True)


@app.function(gpu="H100", timeout=30 * 60)
def speed_probe(preset: str = "1B", opts: str = "baseline", steps: int = 25, warmup: int = 10,
                micro: int = 8, seq_len: int = 1024):
    """Synthetic throughput probe (no data needed): measures ms/step, tokens/s and peak memory
    for one optimization preset at the target model scale. accum=1, random tokens."""
    import os
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # always-on opt; set before torch
    import sys, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from config import get_config, TrainConfig
    from model import MoETransformer, count_params
    from optim import build_optimizers

    opt_sets, orthog = OPT_PRESETS.get(opts, ([], "ns5"))
    dev = torch.device("cuda")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    cfg = get_config(preset)
    for kv in opt_sets:
        k, v = kv.split("=")
        setattr(cfg, k, v.lower() == "true" if v.lower() in ("true", "false") else v)
    cfg.__post_init__()

    model = MoETransformer(cfg).to(dev)
    tc = TrainConfig(orthogonalizer=orthog)
    muon, adamw, _ = build_optimizers(model, tc)
    cmodel = torch.compile(model)
    pc = count_params(model)

    x = torch.randint(0, cfg.vocab_size, (micro, seq_len), device=dev)
    y = torch.randint(0, cfg.vocab_size, (micro, seq_len), device=dev)

    def one_step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = cmodel(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        return loss

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    toks = micro * seq_len
    res = {"opts": opts, "preset": preset, "ms_per_step": dt * 1000,
           "tokens_per_s": toks / dt, "peak_gb": peak_gb,
           "total_M": pc["total"] / 1e6, "micro": micro, "seq_len": seq_len}
    print(f"[{opts:>9} @ {preset}] {dt*1000:7.1f} ms/step | {toks/dt:9.0f} tok/s | "
          f"peak {peak_gb:5.1f} GB | params {pc['total']/1e6:.1f}M", flush=True)
    return res


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
         batch_tokens: int = 262144, save_every: int = 0, data: str = "10B",
         opts: str = "baseline", tasks: str = "", limit: int = 0, fewshot: int = 0,
         init_from: str = "", lr_mult: float = 1.0, gpu_type: str = "H100"):
    data_dir = DATASETS[data][1]
    if action == "prep_data":
        prep_data.remote(tasks or "mmlu")
        return
    if action == "lmeval":
        # EleutherAI lm-evaluation-harness on trained checkpoint(s). run_name="both" -> 130M + 500M.
        targets = ["130M_10B", "500M_40B"] if run_name in ("both", "baseline", "") else [run_name]
        tk = tasks or EVAL_TASKS
        args_t = [(t, "model.pt", tk, limit, 32, fewshot) for t in targets]
        print(f"lm-eval on {targets} | tasks={tk} | {fewshot}-shot | limit={limit or 'full'}", flush=True)
        summaries = {t: s for t, s in zip(targets, lm_eval_run.starmap(args_t))}
        all_tasks = sorted({k for s in summaries.values() for k in s})
        print("\n=== lm-eval COMPARISON (acc / acc_norm) ===", flush=True)
        header = f"{'task':>16} | " + " | ".join(f"{t:>12}" for t in targets)
        print(header); print("-" * len(header))
        for task in all_tasks:
            cells = []
            for t in targets:
                v = summaries[t].get(task)
                cells.append(f"{list(v.values())[0]:>12.4f}" if v else f"{'-':>12}")
            print(f"{task:>16} | " + " | ".join(cells), flush=True)
        for t in targets:
            vals = [list(v.values())[0] for v in summaries[t].values()]
            if vals:
                print(f"  {t} average: {sum(vals)/len(vals):.4f}", flush=True)
        return
    if action == "lmeval_hf":
        # eval reference HF models through OUR harness (same 7-task MicroLlama protocol), then
        # merge with our measured numbers into one comparison table. `overrides` = subset of REF_MODELS keys.
        keys = [k for k in overrides.split(",") if k] if overrides else list(REF_MODELS)
        tk = tasks or EVAL_TASKS_7
        cols = [c for c in tk.split(",") if c]   # task display order
        args_t = [(k, REF_MODELS[k][0], tk, limit, "auto", fewshot) for k in keys]
        print(f"lm-eval (HF) on {len(keys)} models | tasks={tk} | {fewshot}-shot | limit={limit or 'full'}",
              flush=True)
        rows = {}  # name -> {"params","tokens","scores","avg"}
        for k, r in zip(keys, hf_eval.starmap(args_t, return_exceptions=True)):
            if isinstance(r, Exception):
                print(f"  !! {k} FAILED: {type(r).__name__}: {str(r)[:200]}", flush=True)
                continue
            rows[k] = {"params": REF_MODELS[k][1], "tokens": REF_MODELS[k][2],
                       "scores": r["scores"], "avg": r["avg"]}
        # add our measured models
        for name, d in OUR_RESULTS.items():
            rows[name] = d
        # sort by avg (desc), print the full table
        order = sorted(rows, key=lambda n: (rows[n]["avg"] is None, -(rows[n]["avg"] or 0)))
        short = {"hellaswag": "hella", "openbookqa": "obqa", "winogrande": "wino",
                 "arc_challenge": "arc_c", "arc_easy": "arc_e", "boolq": "boolq", "piqa": "piqa"}
        head = (f"{'model':>18} | {'params':>13} | {'tok':>5} | "
                + " | ".join(f"{short.get(c, c):>6}" for c in cols) + f" | {'avg':>6}")
        print("\n=== 7-task comparison (acc_norm; winogrande/boolq=acc; run through our harness) ===", flush=True)
        print(head); print("-" * len(head))
        for n in order:
            d = rows[n]
            cells = " | ".join(f"{d['scores'].get(c, float('nan')):>6.2f}" for c in cols)
            star = " *" if n.startswith("OURS") else ""
            print(f"{n:>18} | {d['params']:>13} | {d['tokens']:>5} | {cells} | "
                  f"{(d['avg'] if d['avg'] is not None else float('nan')):>6.2f}{star}", flush=True)
        print("\n(* = our MoE models)", flush=True)
        return
    if action == "download":
        download.remote(chunks, data)
    elif action == "smoke":
        smoke.remote(preset)
    elif action == "train":
        if gpus == 8:
            fn = train_8b200 if gpu_type.upper() == "B200" else train_8
        else:
            fn = train_1
        print(f"training on {gpus}x{gpu_type.upper() if gpus == 8 else 'H100'}", flush=True)
        fn.remote(preset, steps, run_name, overrides, micro, seq_len, batch_tokens,
                  save_every, data_dir, opts, init_from, lr_mult)
    elif action == "gpu_bench":
        kv = dict(p.split("=", 1) for p in overrides.split(",") if "=" in p) if overrides else {}
        r = gpu_bench.remote(diff_run=(run_name if run_name not in ("baseline", "") else "500M_diff_20b"),
                             ar_run=kv.get("ar_run", "500M_40B"),
                             gen_len=int(kv.get("gen_len", 128)), block=int(kv.get("block", 32)),
                             steps=int(kv.get("steps", 32)))
        print(r, flush=True)
    elif action == "diffgen":
        # iterative-denoising sample from a diffusion checkpoint. run_name=run, overrides="ckpt=ckpt_1000.pt"
        kv = dict(p.split("=", 1) for p in overrides.split(",") if "=" in p) if overrides else {}
        diffgen.remote(run_name=run_name or "500M_diff_5b", ckpt=kv.get("ckpt", "ckpt_1000.pt"),
                       prompt=kv.get("prompt", ""), gen_len=int(kv.get("gen_len", 64)),
                       block=int(kv.get("block", 32)), steps=int(kv.get("steps", 64)),
                       temperature=float(kv.get("temperature", 0.0)),
                       rep_penalty=float(kv.get("rep_penalty", 1.0)),
                       remask_steps=int(kv.get("remask_steps", 0)),
                       remask_frac=float(kv.get("remask_frac", 0.3)),
                       sweep=int(kv.get("sweep", 0)))
    elif action == "speedtest":
        # synthetic throughput probe at the target scale (default 1B); one H100 per variant, parallel.
        variants = ["baseline", "fused_ce", "polar", "all_safe"]  # fp8 dropped: no speedup + zero-grad (see OPT_PRESETS)
        args_t = [(preset, ov, steps if steps != 4000 else 25, 10, micro if micro != 16 else 8, seq_len)
                  for ov in variants]
        print(f"speed probe at {preset} (micro={args_t[0][4]}x{seq_len}) for {len(variants)} variants...",
              flush=True)
        raw = list(speed_probe.starmap(args_t, return_exceptions=True))
        rows = []
        for v, r in zip(variants, raw):
            if isinstance(r, Exception):
                print(f"  !! {v} FAILED: {type(r).__name__}: {r}", flush=True)
            else:
                rows.append(r)
        if not rows:
            raise SystemExit("all speed probes failed")
        base = next((r for r in rows if r["opts"] == "baseline"), None)
        rows.sort(key=lambda r: r["ms_per_step"])
        print(f"\n=== SPEED @ {preset} ===")
        print(f"{'opts':>10} | {'ms/step':>8} | {'tok/s':>9} | {'peak GB':>8} | {'speedup':>8} | {'params M':>9}")
        print("-" * 72)
        for r in rows:
            sp = f"{base['ms_per_step']/r['ms_per_step']:.2f}x" if base else "-"
            print(f"{r['opts']:>10} | {r['ms_per_step']:>8.1f} | {r['tokens_per_s']:>9.0f} | "
                  f"{r['peak_gb']:>8.1f} | {sp:>8} | {r['total_M']:>9.1f}")
    elif action == "ablate_opts":
        # quality ablation: short real-data training at `preset` for each optimization variant.
        variants = ["baseline", "fused_ce", "polar", "all_safe"]  # fp8 dropped: no speedup + zero-grad (see OPT_PRESETS)
        st = steps if steps != 4000 else 800
        args_t = [(preset, st, f"opt_{v}", "", micro, seq_len, batch_tokens, 0, data_dir, v)
                  for v in variants]
        print(f"opt-quality ablation at {preset}, {st} steps each (parallel H100s)...", flush=True)
        done = []
        for res in train_1.starmap(args_t):
            done.append(res)
            print(f"  done: {res.get('run'):>14}  val_loss={res.get('final_val_loss')}", flush=True)
        done.sort(key=lambda r: (r.get("final_val_loss") is None, r.get("final_val_loss")))
        print("\n=== OPT-QUALITY LEADERBOARD (final val loss) ===", flush=True)
        for r in done:
            print(f"{r.get('run'):>14} | {r.get('final_val_loss')}", flush=True)
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
        raise SystemExit(f"unknown action {action!r} (use download|smoke|train|speedtest|ablate_opts|"
                         "ablate|results|generate|specdecode|lmeval|lmeval_hf)")
