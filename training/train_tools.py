"""DDP SFT for tool use (function calling) on the MoE LLM.

Loads the architecture from --backbone, overlays an init checkpoint's LLM weights (e.g. the joint5
unified model), and instruction-tunes on Nemotron tool-call pairs. Loss is masked to the tool-call
completion (handled in tool_data). Text-only — no multimodal encoders.

  torchrun --standalone --nproc_per_node=8 train_tools.py --backbone ... --init ... --train ... --out ...
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path for the `hobbylm` package
from hobbylm.generate import load_model
from hobbylm.tool_data import ToolCallSFT, TrajectorySFT, tool_collate, tool_collate_weighted


def sft_diffusion_mask(ids, tgt, mask_id, eps=1e-3):
    """LLaDA-style SFT noising: mask ONLY the completion (assistant) tokens (tgt != -1) — the prompt
    stays clean and is attended bidirectionally. Returns (noisy, labels, p_mask) for the model's
    diffusion loss (loss only on the masked completion positions, reweighted by 1/p_mask)."""
    B, L = ids.shape
    dev = ids.device
    trainable = tgt != -1
    t = torch.rand(B, device=dev) * (1 - eps) + eps
    p_mask = t[:, None].expand(B, L).contiguous()
    mask = (torch.rand(B, L, device=dev) < p_mask) & trainable
    # guarantee >=1 masked completion token per sequence (so no row contributes a zero/NaN loss)
    none = trainable.any(1) & ~mask.any(1)
    if none.any():
        for r in none.nonzero(as_tuple=True)[0].tolist():
            idxs = trainable[r].nonzero(as_tuple=True)[0]
            mask[r, idxs[torch.randint(len(idxs), (1,), device=dev)]] = True
    noisy = torch.where(mask, torch.full_like(ids, mask_id), ids)
    labels = torch.where(mask, ids, torch.full_like(ids, -1))
    return noisy, labels, p_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--init", default="", help="checkpoint whose ['model'] LLM weights to start from")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--weighted", type=int, default=0, help="1 -> weighted CE (name 3x/value 2x/key 1.5x)")
    ap.add_argument("--traj", type=int, default=0, help="1 -> multi-turn trajectory SFT (loss on all assistant turns)")
    ap.add_argument("--diffusion", type=int, default=0, help="1 -> masked-diffusion (LLaDA) SFT objective")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()
    diffusion = bool(args.diffusion)
    weighted = bool(args.weighted) and not args.traj and not diffusion
    traj = bool(args.traj)

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        dev = torch.device("cuda", local); torch.cuda.set_device(dev)
    else:
        rank, world, local = 0, 1, 0
        dev = torch.device("cuda")
    master = rank == 0

    def log(*a):
        if master:
            print(*a, flush=True)

    torch.manual_seed(1234 + rank); torch.set_float32_matmul_precision("high")
    llm, cfg, vloss, _ = load_model(args.backbone, dev)
    if args.init:
        ck = torch.load(args.init, map_location=dev, weights_only=False)
        llm.load_state_dict(ck["model"])
        log(f"init LLM from {args.init}")
    llm.set_bias_update_rate(0.0)                     # freeze MoE aux-free balancing bias during SFT
    mask_id = getattr(cfg, "mask_token_id", 50257)
    if diffusion:
        assert getattr(cfg, "diffusion", False), "--diffusion needs a diffusion backbone (cfg.diffusion=True)"
        log(f"masked-diffusion SFT | mask_id={mask_id}")
    raw = llm
    model = DDP(llm, device_ids=[local]) if ddp else llm
    n_tr = sum(p.numel() for p in raw.parameters() if p.requires_grad) / 1e6
    log(f"backbone d{cfg.d_model} L{cfg.n_layers} val={vloss:.3f} | trainable={n_tr:.0f}M")

    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.max_steps - args.warmup)))

    ds = TrajectorySFT(args.train, max_len=args.max_len) if traj else ToolCallSFT(args.train, max_len=args.max_len, weighted=weighted)
    smp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=args.micro, sampler=smp, shuffle=(smp is None), num_workers=4,
                    collate_fn=(tool_collate_weighted if weighted else tool_collate),
                    drop_last=True, pin_memory=True, persistent_workers=True)
    log(f"train examples={len(ds)} | micro={args.micro} world={world} | traj={traj} weighted={weighted}")

    def save(tag):
        if not master:
            return
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": raw.state_dict(), "config": {**cfg.to_dict(), "preset": "500M"},
                    "backbone": args.backbone}, f"{args.out}/{tag}")
        log(f"saved -> {args.out}/{tag}")

    model.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step, t0, run, last, done = 0, time.time(), 0.0, float("nan"), False
    while not done:
        if smp is not None:
            smp.set_epoch(step)
        for batch in dl:
            if weighted:
                ids, tgt, wts = batch
                wts = wts.to(dev)
            else:
                ids, tgt = batch
            ids, tgt = ids.to(dev), tgt.to(dev)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            with amp:
                if diffusion:
                    noisy, labels, p_mask = sft_diffusion_mask(ids, tgt, mask_id)
                    loss, _ = model(noisy, targets=labels, p_mask=p_mask)
                elif weighted:
                    logits, aux = model(ids)            # (B,L,V); no internal CE
                    V = logits.shape[-1]
                    ce = F.cross_entropy(logits.reshape(-1, V).float(), tgt.reshape(-1),
                                         ignore_index=-1, reduction="none")
                    w = wts.reshape(-1)
                    loss = (ce * w).sum() / w.sum().clamp_min(1.0) + aux
                else:
                    loss, _ = model(ids, targets=tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
            last = loss.item(); run += last
            if master and step % args.log_every == 0:
                log(f"step {step:5d} | loss {last:.3f} | avg {run/(step+1):.3f} | "
                    f"lr {opt.param_groups[0]['lr']:.2e} | {(time.time()-t0)/(step+1)*1000:.0f}ms/step")
            if args.save_every and (step + 1) % args.save_every == 0:
                save(f"ckpt_{step+1}.pt")
            step += 1
            if step >= args.max_steps:
                done = True; break

    save("model.pt")
    log(f"tool SFT done | final {last:.3f}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
