"""Speculative decoding: small draft model proposes K tokens, large target verifies in one pass.

Uses the standard speculative-sampling accept/reject (Leviathan et al. 2023 / Chen et al. 2023):
the output distribution provably equals the TARGET model's, while emitting multiple tokens per
target forward pass. Draft = 130M, target = 500M (shared GPT-2 vocab).

  python spec_decode.py --draft runs/130M_10B/model.pt --target runs/500M_40B/model.pt
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F

from generate import load_model, GPT2_VALID, EOT


def _probs(logits_row: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """logits_row: (1, vocab) -> proper prob distribution after temp + top-p (fp32)."""
    lg = logits_row.float().clone()
    lg[:, GPT2_VALID:] = -float("inf")
    t = temperature if temperature > 0 else 1.0
    lg = lg / t
    if top_p and top_p < 1.0:
        s, si = torch.sort(lg, descending=True)
        cum = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
        rm = cum > top_p
        rm[..., 1:] = rm[..., :-1].clone()
        rm[..., 0] = False
        lg[0, si[0, rm[0]]] = -float("inf")
    return F.softmax(lg, dim=-1)


@torch.no_grad()
def spec_generate(draft, target, idx, max_new_tokens, K, temperature, top_p, device, ctx_len=1024):
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    n_new = n_accept = n_proposed = n_target_passes = 0
    done = False
    while n_new < max_new_tokens and not done:
        L = idx.shape[1]
        assert L + K <= ctx_len, "demo assumes sequence stays under ctx_len"
        # 1) draft proposes K tokens, recording its sampling distribution q at each step
        d_tokens, d_probs = [], []
        cur = idx
        for _ in range(K):
            with amp:
                dl, _ = draft(cur)
            q = _probs(dl[:, -1, :], temperature, top_p)
            tok = torch.multinomial(q, 1)
            d_tokens.append(tok)
            d_probs.append(q)
            cur = torch.cat([cur, tok], dim=1)
        # 2) target verifies all K in ONE forward pass over [idx + drafts]
        with amp:
            tl, _ = target(cur)
        n_target_passes += 1
        n_proposed += K
        base = L - 1  # tl[:, L-1+j] is target's distribution for the j-th drafted token
        # 3) accept/reject sweep
        for j in range(K):
            p = _probs(tl[:, base + j, :], temperature, top_p)
            q = d_probs[j]
            tok = d_tokens[j]
            tid = tok[0, 0].item()
            ratio = (p[0, tid] / (q[0, tid] + 1e-9)).clamp(max=1.0)
            if torch.rand(1, device=device) < ratio:                 # accept
                idx = torch.cat([idx, tok], dim=1)
                n_new += 1
                n_accept += 1
                if tid == EOT:
                    done = True
                    break
            else:                                                    # reject -> resample residual
                resid = (p - q).clamp(min=0)
                resid = resid / (resid.sum() + 1e-9)
                nt = torch.multinomial(resid, 1)
                idx = torch.cat([idx, nt], dim=1)
                n_new += 1
                if nt[0, 0].item() == EOT:
                    done = True
                break
        else:
            # all K accepted -> free bonus token from target's last position
            p = _probs(tl[:, base + K, :], temperature, top_p)
            nt = torch.multinomial(p, 1)
            idx = torch.cat([idx, nt], dim=1)
            n_new += 1
            if nt[0, 0].item() == EOT:
                done = True
    accept_rate = n_accept / max(1, n_proposed)
    toks_per_pass = n_new / max(1, n_target_passes)
    return idx, dict(accept_rate=accept_rate, target_passes=n_target_passes,
                     toks_per_target_pass=toks_per_pass, new_tokens=n_new)


@torch.no_grad()
def target_only_generate(target, idx, max_new_tokens, temperature, top_p, device, ctx_len=1024):
    """Baseline: plain autoregressive sampling from the target (one target pass per token)."""
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    passes = 0
    for _ in range(max_new_tokens):
        with amp:
            tl, _ = target(idx[:, -ctx_len:])
        passes += 1
        p = _probs(tl[:, -1, :], temperature, top_p)
        nt = torch.multinomial(p, 1)
        idx = torch.cat([idx, nt], dim=1)
        if nt[0, 0].item() == EOT:
            break
    return idx, passes


def run(draft_path, target_path, prompts, max_new_tokens=100, K=4, temperature=0.8, top_p=0.95, device=None):
    import time
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draft, dcfg, dvl, _ = load_model(draft_path, device)
    target, tcfg, tvl, _ = load_model(target_path, device)
    enc = tiktoken.get_encoding("gpt2")
    print(f"draft  {draft_path}: d{dcfg.d_model} L{dcfg.n_layers} val={dvl:.3f}")
    print(f"target {target_path}: d{tcfg.d_model} L{tcfg.n_layers} val={tvl:.3f}")
    print(f"speculative K={K}, temp={temperature}, top_p={top_p}\n", flush=True)

    # warmup (compile/caches) so timing is fair
    warm = torch.tensor([enc.encode_ordinary("The")], device=device)
    spec_generate(draft, target, warm, 8, K, temperature, top_p, device)
    target_only_generate(target, warm, 8, temperature, top_p, device)
    torch.cuda.synchronize() if device.type == "cuda" else None

    for p in prompts:
        ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=device)
        t0 = time.time()
        out, stats = spec_generate(draft, target, ids.clone(), max_new_tokens, K, temperature, top_p, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        spec_t = time.time() - t0

        t0 = time.time()
        base_out, base_passes = target_only_generate(target, ids.clone(), max_new_tokens, temperature, top_p, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        base_t = time.time() - t0

        print("=" * 72)
        print(f"PROMPT: {p!r}")
        print(f"[speculative] {stats['new_tokens']} toks in {spec_t:.2f}s "
              f"({stats['new_tokens']/spec_t:.1f} tok/s) | accept_rate={stats['accept_rate']:.2f} "
              f"| {stats['toks_per_target_pass']:.2f} toks/target-pass ({stats['target_passes']} passes)")
        print(f"[target-only] {base_out.shape[1]-ids.shape[1]} toks in {base_t:.2f}s "
              f"({(base_out.shape[1]-ids.shape[1])/base_t:.1f} tok/s, {base_passes} target passes)")
        print(f"SPEEDUP: {base_t/spec_t:.2f}x")
        print("--- speculative output ---")
        print(enc.decode(out[0].tolist()))
        print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="runs/130M_10B/model.pt")
    ap.add_argument("--target", default="runs/500M_40B/model.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()
    default_prompts = [
        "The meaning of life is",
        "Once upon a time, there was a",
        "In 2023, scientists discovered that",
    ]
    prompts = [args.prompt] if args.prompt else default_prompts
    run(args.draft, args.target, prompts, args.max_new_tokens, args.K, args.temperature, args.top_p)
