"""DPO (Direct Preference Optimization) for the MoE LLM — the alignment phase after chat SFT.

Loads the SFT model as the POLICY and a frozen copy as the REFERENCE, then optimizes the standard DPO
loss on (prompt, chosen, rejected) preference pairs (UltraFeedback): raise logp(chosen) and lower
logp(rejected) relative to the reference. Loss masked to the completion tokens. Same SYSTEM/USER/ASSISTANT
format as SFT, so the aligned model still slots into hobby-chat.

  torchrun --standalone --nproc_per_node=8 train_dpo.py --backbone ... --init <sft.pt> --train pref.jsonl --out ...
"""
from __future__ import annotations
import argparse, copy, json, math, os, time
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import tiktoken
from generate import load_model

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class DPOData(Dataset):
    """Preference pairs -> tokenized (chosen, rejected) with completion masks. Each row:
    {prompt: 'TEXT ending ASSISTANT:', chosen: ' <answer>', rejected: ' <answer>'}."""
    def __init__(self, path, max_len=1024):
        self.rows = []
        for p in path.split(","):
            with open(p, encoding="utf-8") as f:
                self.rows += [json.loads(l) for l in f]
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def _enc(self, prompt, comp):
        p = ENC.encode_ordinary(prompt)
        c = ENC.encode_ordinary(comp) + [EOT]
        ids = (p + c)[: self.max_len]
        mask = ([0] * len(p) + [1] * len(c))[: self.max_len]
        return ids, mask

    def __getitem__(self, i):
        r = self.rows[i]
        ci, cm = self._enc(r["prompt"], r["chosen"])
        ri, rm = self._enc(r["prompt"], r["rejected"])
        return ci, cm, ri, rm


def collate(batch):
    """Pad chosen+rejected of the whole batch to one length -> (ids, mask) for [chosen..., rejected...]."""
    seqs, masks = [], []
    for ci, cm, ri, rm in batch:
        seqs.append(ci); masks.append(cm)
    for ci, cm, ri, rm in batch:
        seqs.append(ri); masks.append(rm)
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), EOT, dtype=torch.long)
    msk = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        msk[i, : len(m)] = torch.tensor(m, dtype=torch.long)
    return ids, msk


def seq_logp(model, ids, mask):
    """Sum of next-token log-probs over the completion (mask) tokens, per sequence."""
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    tok = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:].float()
    return (tok * m).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--init", required=True, help="SFT checkpoint to start policy+reference from")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
        dev = torch.device("cuda", local); torch.cuda.set_device(dev)
    else:
        rank, world, local, dev = 0, 1, 0, torch.device("cuda")
    master = rank == 0
    def log(*a):
        if master: print(*a, flush=True)
    torch.manual_seed(1234 + rank); torch.set_float32_matmul_precision("high")

    policy, cfg, _, _ = load_model(args.backbone, dev)
    ck = torch.load(args.init, map_location=dev, weights_only=False)
    policy.load_state_dict(ck["model"]); log(f"policy+ref init from {args.init}")
    policy.set_bias_update_rate(0.0)
    ref, _, _, _ = load_model(args.backbone, dev)
    ref.load_state_dict(ck["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    raw = policy
    model = DDP(policy, device_ids=[local]) if ddp else policy
    opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - args.warmup) / max(1, args.max_steps - args.warmup)))

    ds = DPOData(args.train, max_len=args.max_len)
    smp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=args.micro, sampler=smp, shuffle=(smp is None), num_workers=4,
                    collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    log(f"DPO pairs={len(ds)} | micro={args.micro} world={world} beta={args.beta} lr={args.lr}")

    def save(tag):
        if not master: return
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": raw.state_dict(), "config": {**cfg.to_dict(), "preset": "500M"},
                    "backbone": args.backbone}, f"{args.out}/{tag}")
        log(f"saved -> {args.out}/{tag}")

    model.train()
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    step, t0, run_acc, done = 0, time.time(), 0.0, False
    while not done:
        if smp is not None:
            smp.set_epoch(step)
        for ids, msk in dl:
            ids, msk = ids.to(dev), msk.to(dev)
            n = ids.shape[0] // 2
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            with amp:
                with torch.no_grad():
                    r_lp = seq_logp(ref, ids, msk)
                p_lp = seq_logp(model, ids, msk)
                pc, pr = p_lp[:n], p_lp[n:]
                rc, rr = r_lp[:n], r_lp[n:]
                logits = args.beta * ((pc - rc) - (pr - rr))
                loss = -F.logsigmoid(logits).mean()
                acc = (logits > 0).float().mean()       # fraction where chosen is preferred
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            opt.step()
            run_acc += acc.item()
            if master and step % args.log_every == 0:
                log(f"step {step:5d} | loss {loss.item():.4f} | acc {acc.item():.3f} | "
                    f"avg_acc {run_acc/(step+1):.3f} | lr {opt.param_groups[0]['lr']:.2e} | "
                    f"{(time.time()-t0)/(step+1)*1000:.0f}ms/step")
            if args.save_every and (step + 1) % args.save_every == 0:
                save(f"ckpt_{step+1}.pt")
            step += 1
            if step >= args.max_steps:
                done = True; break
    save("model.pt")
    log("DPO done")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
