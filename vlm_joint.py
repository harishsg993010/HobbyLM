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
import torch.nn.functional as F
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
# 4-step cycle -> 50% image, 25% video, 25% audio (speech, if enabled, is appended -> 5-way mix)
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
    ap.add_argument("--speech", default="", help="path to speech_projector.pt -> enables the 5th (speech) path")
    ap.add_argument("--va", default="gpt-omni/VoiceAssistant-400K")
    ap.add_argument("--va_shards", type=int, default=4)
    ap.add_argument("--tools", default="", help="path to tools_train.jsonl -> enables the 6th (tool-use) path")
    ap.add_argument("--tools_rep", type=int, default=1, help="repeat single-shot TEXT must-call in the cycle")
    ap.add_argument("--text_traj", default="", help="multi-turn TEXT trajectories -> text query->call->summarize")
    ap.add_argument("--speech_tool", default="", help="spoken-query tool loops -> speech query->call->summarize")
    ap.add_argument("--init_joint", default="", help="continue from a joint ckpt (model + all projectors) -> no regression")
    ap.add_argument("--stream_agentic", type=int, default=0, help="stream nvidia/Nemotron-SFT-Agentic-v2")
    ap.add_argument("--stream_vlm", type=int, default=0, help="stream nvidia/Llama-Nemotron-VLM-Dataset-v1 captioning")
    ap.add_argument("--stream_ocr", type=int, default=0, help="repeat count for the ocr_4 OCR path (0=off)")
    ap.add_argument("--backbone", default=BACKBONE, help="LLM backbone ckpt (provides config incl. rope_theta/ctx)")
    ap.add_argument("--vision_id", default="google/siglip2-so400m-patch14-384", help="SigLIP2 encoder (res)")
    ap.add_argument("--ocr_max_len", type=int, default=2048, help="OCR seq cap (raise at long ctx for full docs)")
    ap.add_argument("--stream_smol", type=int, default=0, help="repeat count for HuggingFaceTB/smoltalk (text)")
    ap.add_argument("--stream_mobile", type=int, default=0, help="repeat count for google/mobile-actions (tool)")
    ap.add_argument("--aria_jsonl", default="", help="pre-staged Aria desktop compact jsonl -> UI grounding path")
    ap.add_argument("--aria_zip", default="", help="pre-staged Aria desktop screenshots zip")
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

    use_speech = bool(args.speech)
    use_tools = bool(args.tools)
    use_ttraj = bool(args.text_traj)
    use_stool = bool(args.speech_tool)
    use_sagent = bool(args.stream_agentic)
    use_svlm = bool(args.stream_vlm)
    use_socr = args.stream_ocr > 0
    use_smol = args.stream_smol > 0
    use_mobile = args.stream_mobile > 0
    use_aria = bool(args.aria_jsonl and args.aria_zip)
    cycle = (CYCLE + (["speech"] if use_speech else []) + (["tools"] * args.tools_rep if use_tools else [])
             + (["text_traj"] if use_ttraj else []) + (["speech_tool"] if use_stool else [])
             + (["stream_agentic"] if use_sagent else []) + (["stream_vlm"] if use_svlm else [])
             + (["stream_ocr"] * args.stream_ocr if use_socr else [])
             + (["stream_smol"] * args.stream_smol if use_smol else [])
             + (["stream_mobile"] * args.stream_mobile if use_mobile else [])
             + (["stream_aria"] if use_aria else []))
    vis = SiglipVision(model_id=args.vision_id, device=dev); vid = SiglipVideo(vis); aud = ClapAudio(device=dev)
    spk = None
    if use_speech:
        from speech import WhisperSpeech
        spk = WhisperSpeech(device=dev)
    llm, cfg, vloss, _ = load_model(args.backbone, dev)
    vlm = MoEVLM(llm, vision_dim=vis.hidden, audio_dim=aud.hidden,
                 speech_dim=(spk.hidden if spk is not None else None)).to(dev)
    if args.init_joint:                                  # continue from a full joint ckpt -> no regression
        jk = torch.load(args.init_joint, map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(jk["model"]); vlm.mm_projector.load_state_dict(jk["projector"])
        vlm.audio_projector.load_state_dict(jk["audio_projector"])
        if use_speech:
            vlm.speech_projector.load_state_dict(jk["speech_projector"])
        log(f"continued from joint ckpt {args.init_joint}")
    else:
        s2 = torch.load(args.stage2, map_location=dev, weights_only=False)
        vlm.llm.load_state_dict(s2["model"]); vlm.mm_projector.load_state_dict(s2["projector"])
        apk = torch.load(args.audio, map_location=dev, weights_only=False)
        vlm.audio_projector.load_state_dict(apk["audio_projector"])
        if use_speech:
            spk_ck = torch.load(args.speech, map_location=dev, weights_only=False)
            vlm.speech_projector.load_state_dict(spk_ck["speech_projector"])
    vlm.set_llm_trainable(True); llm.set_bias_update_rate(0.0)
    log(f"joint init: LLM+mm from {args.stage2}, audio from {args.audio}"
        f"{', speech from ' + args.speech if use_speech else ''} | cycle={cycle} | "
        f"trainable={sum(p.numel() for p in vlm.parameters() if p.requires_grad)/1e6:.0f}M")

    raw = vlm
    if ddp:
        # find_unused_parameters: each step uses only ONE projector (image/video->mm, audio->audio),
        # so the other projector gets no grad that step -> DDP must tolerate unused params.
        vlm = DDP(vlm, device_ids=[local], find_unused_parameters=True)

    proj_ids = {id(p) for p in raw.mm_projector.parameters()} | {id(p) for p in raw.audio_projector.parameters()}
    if use_speech:
        proj_ids |= {id(p) for p in raw.speech_projector.parameters()}
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
    spk_it = None
    if use_speech:
        from vlm_va_data import VoiceAssistantQA, va_collate
        spk_it = mk(VoiceAssistantQA(args.va, max_shards=args.va_shards), va_collate)
    tool_it = None
    if use_tools:
        from tool_data import ToolCallSFT, tool_collate_weighted
        tool_it = mk(ToolCallSFT(args.tools, max_len=2048, weighted=True), tool_collate_weighted)
    ttraj_it = None
    if use_ttraj:
        from tool_data import TrajectorySFT, tool_collate as _tc
        ttraj_it = mk(TrajectorySFT(args.text_traj, max_len=2048), _tc)
    stool_it = None
    if use_stool:
        from speech_tool_data import SpeechToolSFT, speech_tool_collate
        stool_it = mk(SpeechToolSFT(args.speech_tool, max_len=2048), speech_tool_collate)

    def mk_stream(ds, coll, workers, mult=1):           # IterableDataset self-shards by rank/worker (no sampler)
        dl = DataLoader(ds, batch_size=args.micro * mult, num_workers=workers, collate_fn=coll,
                        drop_last=True, pin_memory=True, persistent_workers=True)
        return iter(dl)                                  # plain iter: a stall crashes fast (frees GPUs) vs frozen-alive
    sagent_it = svlm_it = None
    if use_sagent:
        from stream_data import StreamAgentic, stream_collate_text
        sagent_it = mk_stream(StreamAgentic(rank=rank, world=world), stream_collate_text, workers=2)
    if use_svlm:
        from stream_data import StreamVLM, stream_collate_vlm
        svlm_it = mk_stream(StreamVLM(rank=rank, world=world), stream_collate_vlm, workers=4)
    socr_it = None
    if use_socr:
        from stream_data import StreamOcr, stream_collate_vlm
        socr_it = mk_stream(StreamOcr(rank=rank, world=world, max_len=args.ocr_max_len), stream_collate_vlm, workers=2)
    smol_it = mobile_it = aria_it = None
    if use_smol:
        from stream_data import StreamSmolTalk, stream_collate_text
        smol_it = mk_stream(StreamSmolTalk(rank=rank, world=world), stream_collate_text, workers=2)
    if use_mobile:
        from stream_data import StreamMobileActions, stream_collate_text
        mobile_it = mk_stream(StreamMobileActions(rank=rank, world=world), stream_collate_text, workers=2)
    if use_aria:
        from stream_data import StreamAria, stream_collate_vlm
        aria_it = mk_stream(StreamAria(args.aria_jsonl, args.aria_zip, rank=rank, world=world),
                            stream_collate_vlm, workers=2)
    log(f"micro={args.micro} world={world} | cycle={cycle}")

    def save(tag):
        if not master:
            return
        os.makedirs(args.out, exist_ok=True)
        blob = {"model": raw.llm.state_dict(), "projector": raw.mm_projector.state_dict(),
                "audio_projector": raw.audio_projector.state_dict(),
                "config": {**cfg.to_dict(), "preset": "500M"}, "vision_dim": vis.hidden,
                "audio_dim": aud.hidden, "backbone": args.backbone}
        if use_speech:
            blob["speech_projector"] = raw.speech_projector.state_dict()
            blob["speech_dim"] = spk.hidden
        torch.save(blob, f"{args.out}/{tag}")
        log(f"saved -> {args.out}/{tag}")

    vlm.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    t0 = time.time()
    _keys = ("image", "video", "audio", "speech", "tools", "text_traj", "speech_tool",
             "stream_agentic", "stream_vlm", "stream_ocr", "stream_smol", "stream_mobile", "stream_aria")
    run = {k: 0.0 for k in _keys}
    cnt = {k: 0 for k in _keys}
    for step in range(args.max_steps):
        modality = cycle[step % len(cycle)]
        m = lr_at(step)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * m

        kw = {}
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
        elif modality == "speech":
            wavs, ids, tgt = next(spk_it)
            with torch.no_grad(), amp:
                feats = spk.encode(wavs)
            kw = {"speech_features": feats}
        elif modality == "tools":
            ids, tgt, wts = next(tool_it)            # text-only; weighted CE (name/value emphasis)
            wts = wts.to(dev)
        elif modality == "text_traj":
            ids, tgt = next(ttraj_it)                # multi-turn TEXT: text query -> call -> summarize
        elif modality == "speech_tool":
            wavs, ids, tgt = next(stool_it)          # SPOKEN query -> call -> summarize
            with torch.no_grad(), amp:
                feats = spk.encode(wavs)
            kw = {"speech_features": feats}
        elif modality == "stream_agentic":
            ids, tgt = next(sagent_it)               # streamed nvidia agentic-v2 (text)
        elif modality == "stream_vlm":
            imgs, ids, tgt = next(svlm_it)           # streamed nvidia VLM captioning (OpenImages S3)
            with torch.no_grad(), amp:
                feats = vis.encode(imgs)
            kw = {"image_features": feats}
        elif modality == "stream_ocr":
            imgs, ids, tgt = next(socr_it)           # streamed nvidia VLM ocr_4 (local pre-staged tars)
            with torch.no_grad(), amp:
                feats = vis.encode(imgs)
            kw = {"image_features": feats}
        elif modality == "stream_smol":
            ids, tgt = next(smol_it)                 # HuggingFaceTB/smoltalk (text chat)
        elif modality == "stream_mobile":
            ids, tgt = next(mobile_it)               # google/mobile-actions (mobile tool-calling, text)
        elif modality == "stream_aria":
            imgs, ids, tgt = next(aria_it)           # Aria-UI desktop grounding (screenshot -> click point)
            with torch.no_grad(), amp:
                feats = vis.encode(imgs)
            kw = {"image_features": feats}
        else:  # audio
            wavs, ids, tgt = next(aud_it)
            with torch.no_grad(), amp:
                feats = aud.encode(wavs)
            kw = {"audio_features": feats}

        ids, tgt = ids.to(dev), tgt.to(dev)
        with amp:
            if modality == "tools":
                logits, aux = vlm(ids)              # (B,L,V) — no modality features, no internal CE
                V = logits.shape[-1]
                ce = F.cross_entropy(logits.reshape(-1, V).float(), tgt.reshape(-1),
                                     ignore_index=-1, reduction="none")
                w = wts.reshape(-1)
                loss = (ce * w).sum() / w.sum().clamp_min(1.0) + aux
            else:
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
