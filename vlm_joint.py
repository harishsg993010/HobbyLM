"""Joint multimodal SFT (DDP): co-train LLM + mm_projector (image & video) + audio_projector on an
interleaved mix of image / video / audio steps. Encoders (SigLIP2, CLAP) frozen.

Video uses the IMAGE data presented as 2-frame clips through VIDEO_TOKEN (no separate video corpus),
so all three sentinels get trained and the LLM co-adapts to all modalities.

  torchrun --standalone --nproc_per_node=8 vlm_joint.py --stage2 ... --audio ... --json ... --zip ... \
      --clotho CLAPv2/Clotho --out ...
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from audio import ClapAudio
from generate import load_model
from multimodal import MoEVLM, IMAGE_TOKEN, VIDEO_TOKEN
from vision import SiglipVision
from video import SiglipVideo
from vlm_data import LlavaSFT, collate
from vlm_audio_data import ClothoAudio, audio_collate

BACKBONE = "/data/runs/500M_ctx2048/model.pt"
# 4-step cycle -> 50% image, 25% video, 25% audio
CYCLE = ["image", "image", "video", "audio"]


def cyc(loader, sampler):
    ep = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(ep)
        for b in loader:
            yield b
        ep += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--clotho", default="CLAPv2/Clotho")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=1600)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--proj_lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=400)
    args = ap.parse_args()

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

    vis = SiglipVision(device=dev); vid = SiglipVideo(vis); aud = ClapAudio(device=dev)
    llm, cfg, vloss, _ = load_model(BACKBONE, dev)
    vlm = MoEVLM(llm, vision_dim=vis.hidden, audio_dim=aud.hidden).to(dev)
    s2 = torch.load(args.stage2, map_location=dev, weights_only=False)
    vlm.llm.load_state_dict(s2["model"]); vlm.mm_projector.load_state_dict(s2["projector"])
    apk = torch.load(args.audio, map_location=dev, weights_only=False)
    vlm.audio_projector.load_state_dict(apk["audio_projector"])
    vlm.set_llm_trainable(True); llm.set_bias_update_rate(0.0)
    log(f"joint init: LLM+mm from {args.stage2}, audio from {args.audio} | "
        f"trainable={sum(p.numel() for p in vlm.parameters() if p.requires_grad)/1e6:.0f}M")

    raw = vlm
    if ddp:
        vlm = DDP(vlm, device_ids=[local])

    proj_ids = {id(p) for p in raw.mm_projector.parameters()} | {id(p) for p in raw.audio_projector.parameters()}
    proj = [p for p in raw.parameters() if p.requires_grad and id(p) in proj_ids]
    llmp = [p for p in raw.parameters() if p.requires_grad and id(p) not in proj_ids]
    opt = torch.optim.AdamW([{"params": llmp, "lr": args.lr}, {"params": proj, "lr": args.proj_lr}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    base_lrs = [args.lr, args.proj_lr]

    def lr_at(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.max_steps - args.warmup)))

    def mk(ds, coll, mult=1):
        smp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
        dl = DataLoader(ds, batch_size=args.micro * mult, sampler=smp, shuffle=(smp is None),
                        num_workers=4, collate_fn=coll, drop_last=True, pin_memory=True, persistent_workers=True)
        return cyc(dl, smp)

    img_it = mk(LlavaSFT(args.json, args.zip, sentinel=IMAGE_TOKEN), collate)
    vid_it = mk(LlavaSFT(args.json, args.zip, sentinel=VIDEO_TOKEN), collate)
    aud_it = mk(ClothoAudio(args.clotho), audio_collate, mult=2)        # audio seq short -> bigger batch
    log(f"micro={args.micro} world={world} | cycle={CYCLE}")

    def save(tag):
        if not master:
            return
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": raw.llm.state_dict(), "projector": raw.mm_projector.state_dict(),
                    "audio_projector": raw.audio_projector.state_dict(),
                    "config": {**cfg.to_dict(), "preset": "500M"}, "vision_dim": vis.hidden,
                    "audio_dim": aud.hidden, "backbone": BACKBONE}, f"{args.out}/{tag}")
        log(f"saved -> {args.out}/{tag}")

    vlm.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    t0, run = time.time(), {"image": 0.0, "video": 0.0, "audio": 0.0}
    cnt = {"image": 0, "video": 0, "audio": 0}
    for step in range(args.max_steps):
        modality = CYCLE[step % len(CYCLE)]
        m = lr_at(step)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * m

        if modality == "image":
            imgs, ids, tgt = next(img_it)
            with torch.no_grad(), amp:
                feats = vis.encode(imgs)
            kw = {"image_features": feats}
        elif modality == "video":
            imgs, ids, tgt = next(vid_it)
            with torch.no_grad(), amp:
                frames = [im for im in imgs for _ in range(2)]          # 2 frames per sample
                f = vis.encode(frames)                                  # (2B,729,C)
                feats = f.reshape(len(imgs), 2 * f.shape[1], f.shape[2])
            kw = {"video_features": feats}
        else:  # audio
            wavs, ids, tgt = next(aud_it)
            with torch.no_grad(), amp:
                feats = aud.encode(wavs)
            kw = {"audio_features": feats}

        ids, tgt = ids.to(dev), tgt.to(dev)
        with amp:
            loss, _ = vlm(ids, targets=tgt, **kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad], 1.0)
        opt.step()
        run[modality] += loss.item(); cnt[modality] += 1
        if master and step % args.log_every == 0:
            avgs = " ".join(f"{k}={run[k]/max(1,cnt[k]):.3f}" for k in run)
            log(f"step {step:5d} | {modality:5s} loss {loss.item():.3f} | avg[{avgs}] | "
                f"lr {opt.param_groups[0]['lr']:.2e} | {(time.time()-t0)/(step+1)*1000:.0f}ms/step")
        if args.save_every and (step + 1) % args.save_every == 0:
            save(f"ckpt_{step+1}.pt")

    save("model.pt")
    log(f"joint done | counts={cnt}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
