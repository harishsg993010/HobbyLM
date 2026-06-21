"""Inference / text generation from a trained moe-lab checkpoint.

Loads a saved checkpoint (model.pt / ckpt_*.pt), rebuilds the model from its embedded config,
and autoregressively samples text with the GPT-2 (tiktoken) tokenizer.

  python generate.py --ckpt runs/130M_10B/model.pt --prompt "The meaning of life is"
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import MoETransformer

GPT2_VALID = 50257          # real GPT-2 tokens (rest of vocab is padding)
EOT = 50256                 # <|endoftext|>


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_d = dict(ck["config"])
    cfg_d.pop("preset", None)
    cfg = ModelConfig(**cfg_d)
    cfg.expert_backend = "grouped" if device.type == "cuda" else "bmm"
    model = MoETransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("val_loss"), ck.get("step")


def _banned_ngram_tokens(prev: list[int], n: int) -> list[int]:
    """Tokens that would complete an already-seen n-gram (no-repeat-ngram blocking)."""
    if n <= 0 or len(prev) < n:
        return []
    seen: dict[tuple, list[int]] = {}
    for i in range(len(prev) - n + 1):
        ng = tuple(prev[i:i + n])
        seen.setdefault(ng[:-1], []).append(ng[-1])
    return seen.get(tuple(prev[-(n - 1):]), [])


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, device,
             top_p=0.95, repetition_penalty=1.3, no_repeat_ngram_size=3, ctx_len=1024):
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ctx_len:]
        with amp:
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float()
        logits[:, GPT2_VALID:] = -float("inf")          # never emit padding tokens

        seq = idx[0].tolist()
        # repetition penalty (CTRL-style): damp logits of already-generated tokens
        if repetition_penalty and repetition_penalty != 1.0:
            uniq = torch.tensor(sorted(set(seq)), device=logits.device)
            lg = logits[0, uniq]
            logits[0, uniq] = torch.where(lg > 0, lg / repetition_penalty, lg * repetition_penalty)
        # no-repeat n-gram blocking
        for t in _banned_ngram_tokens(seq, no_repeat_ngram_size):
            logits[0, t] = -float("inf")

        if temperature > 0:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p and top_p < 1.0:                    # nucleus filtering
                s_logits, s_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                rm = cum > top_p
                rm[..., 1:] = rm[..., :-1].clone()
                rm[..., 0] = False
                logits[0, s_idx[0, rm[0]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        else:
            nxt = logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if nxt.item() == EOT:
            break
    return idx


def run(ckpt_path, prompts, max_new_tokens=120, temperature=0.9, top_k=0, device=None,
        top_p=0.95, repetition_penalty=1.3, no_repeat_ngram_size=3):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, val_loss, step = load_model(ckpt_path, device)
    enc = tiktoken.get_encoding("gpt2")
    print(f"loaded {ckpt_path} | step={step} val_loss={val_loss} | "
          f"d_model={cfg.d_model} layers={cfg.n_layers} experts={cfg.n_experts}/top{cfg.top_k}", flush=True)
    print(f"sampling: temp={temperature} top_k={top_k} top_p={top_p} "
          f"rep_penalty={repetition_penalty} no_repeat_ngram={no_repeat_ngram_size}\n", flush=True)
    for p in prompts:
        ids = torch.tensor([enc.encode_ordinary(p)], dtype=torch.long, device=device)
        out = generate(model, ids, max_new_tokens, temperature, top_k, device,
                       top_p=top_p, repetition_penalty=repetition_penalty,
                       no_repeat_ngram_size=no_repeat_ngram_size)
        text = enc.decode(out[0].tolist())
        print("=" * 70)
        print(f"PROMPT: {p!r}")
        print(text)
        print(flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/130M_10B/model.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.3)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = ap.parse_args()
    default_prompts = [
        "The meaning of life is",
        "Once upon a time, there was a",
        "The capital of France is",
        "In 2023, scientists discovered that",
        "To make a good cup of coffee, you",
        "The most important thing about climate change is",
    ]
    prompts = [args.prompt] if args.prompt else default_prompts
    run(args.ckpt, prompts, args.max_new_tokens, args.temperature, args.top_k,
        top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size)
