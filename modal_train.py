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
    .pip_install("torch==2.12.0", "numpy", "tiktoken", "lm-eval==0.4.9.1", "datasets", "huggingface-hub")
    .add_local_dir(".", "/root/moe-lab")
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
                save_every=0, data_dir=DATA_DIR, opts="baseline"):
    import os
    os.chdir("/root/moe-lab")
    opt_sets, orthog = OPT_PRESETS.get(opts, ([], "ns5"))
    user_sets = overrides.split(",") if overrides else []
    all_sets = [*opt_sets, *user_sets]
    over = ["--set", *all_sets] if all_sets else []
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={gpus}", "train.py",
        "--preset", preset, "--run_name", run_name, "--data_dir", data_dir,
        "--out_dir", "/data/runs", "--max_steps", str(steps), "--micro_batch_seqs", str(micro),
        "--seq_len", str(seq_len), "--batch_tokens", str(batch_tokens),
        "--orthogonalizer", orthog, "--save_every", str(save_every), *over,
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
def train_1(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0,
            data_dir=DATA_DIR, opts="baseline"):
    return _train_body(preset, steps, run_name, overrides, 1, micro, seq_len, batch_tokens,
                       save_every, data_dir, opts)


@app.function(gpu="H100:8", volumes={"/data": vol}, timeout=24 * 60 * 60)
def train_8(preset, steps, run_name, overrides, micro, seq_len, batch_tokens, save_every=0,
            data_dir=DATA_DIR, opts="baseline"):
    return _train_body(preset, steps, run_name, overrides, 8, micro, seq_len, batch_tokens,
                       save_every, data_dir, opts)


# default benchmark suite for small models (all loglikelihood / multiple-choice, GPT-2/Pythia-style)
EVAL_TASKS = "lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,sciq,boolq"


@app.function(image=eval_image, gpu="H100", volumes={"/data": vol},
              timeout=4 * 60 * 60, secrets=[modal.Secret.from_name("huggingface")])
def lm_eval_run(run_name: str, ckpt: str = "model.pt", tasks: str = EVAL_TASKS,
                limit: int = 0, batch_size: int = 32, num_fewshot: int = 0):
    """Run the EleutherAI lm-evaluation-harness on a trained checkpoint."""
    import os, sys, json
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import torch
    from lm_eval import simple_evaluate
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
    print(f"\n=== lm-eval [{run_name}] ({num_fewshot}-shot, limit={limit or 'full'}) ===", flush=True)
    print(f"{'task':>16} | {'metric':>10} | {'value':>7} | {'stderr':>7}")
    print("-" * 50)
    summary = {}
    for task, metrics in sorted(res["results"].items()):
        # prefer acc_norm where present, else acc, else first metric
        pick = None
        for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
            if key in metrics:
                pick = key
                break
        if pick is None:
            pick = next((k for k in metrics if not k.endswith("_stderr") and k != "alias"), None)
        if pick is None:
            continue
        val = metrics[pick]
        stderr = metrics.get(pick.replace(",none", "_stderr,none"), metrics.get(pick + "_stderr", float("nan")))
        base = pick.split(",")[0]
        summary[task] = {base: val}
        try:
            print(f"{task:>16} | {base:>10} | {val:>7.4f} | {float(stderr):>7.4f}", flush=True)
        except (ValueError, TypeError):
            print(f"{task:>16} | {base:>10} | {val:>7.4f} |     n/a", flush=True)
    accs = [list(v.values())[0] for v in summary.values()]
    if accs:
        print("-" * 50)
        print(f"{'AVERAGE':>16} | {'':>10} | {sum(accs)/len(accs):>7.4f} |", flush=True)

    out = {"run": run_name, "step": step, "val_loss": val_loss, "num_fewshot": num_fewshot,
           "limit": limit, "results": res["results"]}
    try:
        with open(f"/data/runs/{run_name}/lm_eval.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        vol.commit()
    except Exception as e:
        print(f"(could not save lm_eval.json: {e})", flush=True)
    return summary


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
         opts: str = "baseline", tasks: str = "", limit: int = 0, fewshot: int = 0):
    data_dir = DATASETS[data][1]
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
    if action == "download":
        download.remote(chunks, data)
    elif action == "smoke":
        smoke.remote(preset)
    elif action == "train":
        fn = train_8 if gpus == 8 else train_1
        fn.remote(preset, steps, run_name, overrides, micro, seq_len, batch_tokens,
                  save_every, data_dir, opts)
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
                         "ablate|results|generate|specdecode|lmeval)")
