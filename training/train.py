"""Training loop for the MoE lab. Single-GPU or DDP (torchrun) on Modal.

  python train.py --preset 130M --max_steps 4000 --run_name baseline
  torchrun --nproc_per_node=8 train.py --preset 1B ...

Ablations override config fields via --set key=value (e.g. --set gating=softmax n_shared=1).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

# Reduce allocator fragmentation (lets us fit a larger batch); must be set before importing torch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path for the `hobbylm` package
from hobbylm.config import TrainConfig, get_config
from hobbylm.data import data_generator, CUDAPrefetcher
from hobbylm.model import MoETransformer, count_params
from hobbylm.optim import build_optimizers
from hobbylm.diffusion import forward_mask


def lr_mult(step: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return (step + 1) / tc.warmup_steps
    cd_start = int(tc.max_steps * (1 - tc.cooldown_frac))
    if step < cd_start:
        return 1.0
    t = (step - cd_start) / max(1, tc.max_steps - cd_start)
    return 1.0 + t * (tc.final_lr_frac - 1.0)


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif v.replace(".", "", 1).replace("-", "", 1).isdigit():
            out[k] = float(v) if "." in v else int(v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="130M")
    ap.add_argument("--run_name", default="baseline")
    ap.add_argument("--data_dir", default="data/fineweb10B")
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_tokens", type=int, default=256 * 1024)
    ap.add_argument("--micro_batch_seqs", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--val_tokens", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--save_every", type=int, default=0, help="save a checkpoint every N steps (0=only final)")
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--orthogonalizer", default="ns5", choices=["ns5", "polar"],
                    help="Muon orthogonalizer: ns5 (Newton-Schulz) or polar (Polar Express)")
    ap.add_argument("--init_from", default="", help="checkpoint .pt to resume model weights from (continued pretrain)")
    ap.add_argument("--resume_from", default="", help="checkpoint .pt to fully resume training state from")
    ap.add_argument("--lr_mult", type=float, default=1.0, help="multiply base LRs (use <1 for finetune/ctx-extension)")
    ap.add_argument("--set", nargs="*", default=[], help="model config overrides key=value")
    args = ap.parse_args()

    # ---- DDP setup ----
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0

    def log(*a):
        if master:
            print(*a, flush=True)

    tc = TrainConfig(seq_len=args.seq_len, batch_tokens=args.batch_tokens,
                     micro_batch_seqs=args.micro_batch_seqs, max_steps=args.max_steps,
                     val_every=args.val_every, val_tokens=args.val_tokens, run_name=args.run_name,
                     out_dir=args.out_dir, compile=not args.no_compile,
                     orthogonalizer=args.orthogonalizer)
    torch.manual_seed(tc.seed + rank)
    torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls (router/embed)

    # auto sqrt-scale LR vs the ablation reference batch (262144 tokens), so larger
    # multi-GPU batches stay principled without per-run retuning.
    REF_BATCH = 262144
    lr_scale = (tc.batch_tokens / REF_BATCH) ** 0.5 * args.lr_mult
    tc.muon_lr *= lr_scale
    tc.adam_lr *= lr_scale

    # ---- model ----
    cfg = get_config(args.preset)
    for k, v in parse_overrides(args.set).items():
        setattr(cfg, k, v)
    cfg.__post_init__()
    if device.type != "cuda":
        cfg.expert_backend = "bmm"   # grouped_mm needs CUDA bf16

    model = MoETransformer(cfg).to(device)   # fp32 master weights; bf16 via autocast
    resume_ck = None
    start_step = 0
    if args.resume_from:
        resume_ck = torch.load(args.resume_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(resume_ck["model"], strict=False)
        start_step = int(resume_ck.get("step", 0) or 0)
        log(f"fully resumed from {args.resume_from} "
            f"(prev step={start_step}, val={resume_ck.get('val_loss')}; "
            f"missing={len(missing)} unexpected={len(unexpected)})")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        log(f"resumed weights from {args.init_from} "
            f"(prev step={ck.get('step')}, val={ck.get('val_loss')}; "
            f"missing={len(missing)} unexpected={len(unexpected)})")
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and tc.bf16 else nullcontext())
    pc = count_params(model)
    log(f"[{args.preset}] total={pc['total']/1e6:.1f}M active={pc['active']/1e6:.1f}M "
        f"({pc['active_pct']:.1f}%)  overrides={parse_overrides(args.set)}")

    raw_model = model
    if tc.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    muon, adamw, (nm, na) = build_optimizers(raw_model, tc)
    log(f"optimizers: Muon over {nm} tensors, AdamW over {na} tensors")
    if resume_ck is not None:
        if "muon" in resume_ck and "adamw" in resume_ck:
            muon.load_state_dict(resume_ck["muon"])
            adamw.load_state_dict(resume_ck["adamw"])
            log("restored optimizer states (Muon + AdamW)")
        else:
            log("resume checkpoint has no optimizer state; continuing with fresh optimizers")
        if "cpu_rng_state" in resume_ck:
            torch.set_rng_state(resume_ck["cpu_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in resume_ck:
            try:
                torch.cuda.set_rng_state(resume_ck["cuda_rng_state"].cpu(), device=device)
            except Exception as e:
                log(f"(cuda rng restore skipped: {e})")

    # ---- data ----
    B, S = tc.micro_batch_seqs, tc.seq_len
    tokens_per_micro = B * S * world
    accum = max(1, tc.batch_tokens // tokens_per_micro)
    train_gen = data_generator(str(Path(args.data_dir) / "fineweb_train_*.bin"), B, S, device,
                               rank, world, to_device=False)
    train_prefetch = CUDAPrefetcher(train_gen, device)   # overlaps H2D copy with compute
    val_pattern = str(Path(args.data_dir) / "fineweb_val_*.bin")
    log(f"batch_tokens={tc.batch_tokens} micro=({B}x{S})x{world} accum={accum} "
        f"lr_scale={lr_scale:.2f} muon_lr={tc.muon_lr:.4f} adam_lr={tc.adam_lr:.2e}")

    out_dir = Path(tc.out_dir) / tc.run_name
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "preset": args.preset}, indent=2))

    def save_ckpt(fname, **extra):
        if not master:
            return
        torch.save({"model": raw_model.state_dict(),
                    "config": {**cfg.to_dict(), "preset": args.preset},
                    "muon": muon.state_dict(),
                    "adamw": adamw.state_dict(),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state(device=device) if device.type == "cuda" else None),
                    **extra}, out_dir / fname)
        log(f"saved checkpoint -> {out_dir / fname}")

    def model_loss(m, x, y):
        # diffusion: ignore the AR-shifted y; mask x in-place and score the masked positions.
        if cfg.diffusion:
            noisy, labels, p_mask = forward_mask(x, cfg.mask_token_id, cfg.mask_eps)
            return m(noisy, labels, p_mask=p_mask)
        return m(x, y)

    @torch.no_grad()
    def evaluate(max_tokens=tc.val_tokens):
        model.eval()
        gen = data_generator(val_pattern, B, S, device, rank, world)
        tot_loss, tot_tok, steps = 0.0, 0, max(1, max_tokens // (B * S * world))
        for _ in range(steps):
            x, y = next(gen)
            with amp:
                loss, _ = model_loss(raw_model, x, y)
            tot_loss += loss.item() * x.numel()
            tot_tok += x.numel()
        model.train()
        t = torch.tensor([tot_loss, tot_tok], device=device)
        if ddp:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return (t[0] / t[1]).item()

    # ---- train ----
    model.train()
    t0 = time.time()
    freeze_step = int(tc.max_steps * tc.bias_anneal_frac)
    if start_step >= freeze_step:
        raw_model.set_bias_update_rate(0.0)
        log(f"resume starts after bias-freeze point ({freeze_step}); keeping aux-free expert bias frozen")
    if start_step >= tc.max_steps:
        raise SystemExit(f"resume checkpoint step {start_step} is already >= target max_steps {tc.max_steps}")
    for step in range(start_step, tc.max_steps):
        # lr schedule
        m = lr_mult(step, tc)
        for g in muon.param_groups:
            g["lr"] = tc.muon_lr * m
        for g in adamw.param_groups:
            g["lr"] = tc.adam_lr * m
        # bias anneal
        if step == freeze_step:
            raw_model.set_bias_update_rate(0.0)
            log(f"step {step}: froze aux-free expert bias")

        # accumulate the loss on-device; only sync to host when we actually log (avoids a
        # device->host stall every micro-step that would serialize the accumulation loop).
        loss_accum = torch.zeros((), device=device)
        for micro in range(accum):
            x, y = train_prefetch.next()
            sync_ctx = model.no_sync() if (ddp and micro < accum - 1) else nullcontext()
            with sync_ctx:
                with amp:
                    loss, _ = model_loss(model, x, y)
                (loss / accum).backward()
            loss_accum += loss.detach() / accum

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        if ddp:
            raw_model.sync_expert_bias()   # keep aux-free bias buffers identical across ranks

        if step % tc.log_every == 0:
            dt = (time.time() - t0) / (step - start_step + 1)
            log(f"step {step:5d} | loss {loss_accum.item():.4f} | lr {tc.muon_lr*m:.4f} | {dt*1000:.0f}ms/step")
        if tc.val_every and (step + 1) % tc.val_every == 0:
            vl = evaluate()
            log(f"  >> val loss {vl:.4f} @ step {step+1}")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(f"ckpt_{step+1}.pt", step=step + 1)

    vl = evaluate()
    log(f"=== final val loss {vl:.4f} ===")
    if master:
        (out_dir / "result.json").write_text(json.dumps({"final_val_loss": vl, "steps": tc.max_steps}))
    save_ckpt("model.pt", step=tc.max_steps, val_loss=vl)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
