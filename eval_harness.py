"""EleutherAI lm-evaluation-harness wrapper for moe-lab MoETransformer checkpoints.

Implements the three LM methods (loglikelihood / loglikelihood_rolling / generate_until)
over our custom MoE model + GPT-2 (tiktoken) tokenizer. Runs on Modal (see modal_train.py
`--action lmeval`). Imported only where lm_eval is installed.

Scoring is standard: logits at position k predict token k+1; right-padding is safe under causal
attention (real tokens never attend to future pads). Padding-vocab columns (>=50257) are masked.
"""
from __future__ import annotations

from contextlib import nullcontext

import tiktoken
import torch
import torch.nn.functional as F

from lm_eval.api.model import LM

from generate import GPT2_VALID, EOT


class MoELMWrapper(LM):
    def __init__(self, model, device, max_length: int = 1024, batch_size: int = 32):
        super().__init__()
        self.model = model
        self._device = device
        self._max_length = max_length
        self.batch_size = batch_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = EOT
        self.amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext())

    # ---- tokenization ----
    def tok_encode(self, s: str) -> list[int]:
        return self.enc.encode_ordinary(s)

    def tok_decode(self, toks: list[int]) -> str:
        return self.enc.decode(toks)

    def _encode_pair(self, context: str, continuation: str):
        """Match lm-eval's TemplateLM._encode_pair: joint-encode so BPE merges at the
        context/continuation boundary are respected, and move trailing context whitespace
        into the continuation. (Encoding the two halves separately mis-tokenizes the seam.)"""
        if context == "":
            return [self.eot], self.tok_encode(continuation)
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tok_encode(context + continuation)
        ctx_enc = self.tok_encode(context)
        return ctx_enc, whole[len(ctx_enc):]

    @property
    def eot_token_id(self) -> int:
        return self.eot

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    # ---- core: batched forward returning RAW logits (B, Lmax, vocab), right-padded ----
    @torch.no_grad()
    def _logits(self, batch_inputs: list[list[int]]) -> torch.Tensor:
        """Forward a right-padded batch; only real positions are meaningful (causal attention).
        Returns raw logits in the model dtype — we log_softmax only the small continuation slice
        per request (not the whole (B,Lmax,vocab)), which is ~100x less softmax compute on mmlu."""
        Lmax = max(len(x) for x in batch_inputs)
        B = len(batch_inputs)
        idx = torch.zeros(B, Lmax, dtype=torch.long, device=self._device)
        for j, x in enumerate(batch_inputs):
            idx[j, :len(x)] = torch.tensor(x, dtype=torch.long, device=self._device)
        with self.amp:
            logits, _ = self.model(idx)
        return logits

    def _logp_slice(self, logits_slice: torch.Tensor) -> torch.Tensor:
        """fp32 log-softmax over the valid vocab for a small (..., vocab) slice."""
        sl = logits_slice.float()
        sl[..., GPT2_VALID:] = -float("inf")       # never score padding-vocab tokens
        return F.log_softmax(sl, dim=-1)

    # ---- loglikelihood: P(continuation | context) ----
    @torch.no_grad()
    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        reqs = []
        for r in requests:
            ctx, cont = r.args
            ctx_enc, cont_enc = self._encode_pair(ctx, cont)
            reqs.append((ctx_enc, cont_enc))

        results: list[tuple[float, bool]] = [(0.0, False)] * len(reqs)
        # sort longest-first so each padded batch wastes the least
        order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i][0]) + len(reqs[i][1])))

        for s in range(0, len(order), self.batch_size):
            chunk = order[s:s + self.batch_size]
            inputs, metas = [], []
            for i in chunk:
                ctx_enc, cont_enc = reqs[i]
                full = ctx_enc + cont_enc
                inp = full[:-1][-self._max_length:]              # predict each next token
                cont_len = min(len(cont_enc), len(inp))
                inputs.append(inp)
                metas.append((i, cont_len, cont_enc))
            logits = self._logits(inputs)
            # compute per-request (ll, greedy) on-GPU, then sync ONCE per batch (not per request).
            # per-request .item() was ~2 GPU->CPU syncs each -> pathologically slow on big tasks (mmlu).
            lls, greedys, idxs = [], [], []
            for j, (i, cont_len, cont_enc) in enumerate(metas):
                if cont_len == 0:
                    results[i] = (0.0, True)
                    continue
                Lj = len(inputs[j])
                sl = self._logp_slice(logits[j, Lj - cont_len:Lj, :])   # (cont_len, vocab) log-probs
                tgt = torch.tensor(cont_enc[len(cont_enc) - cont_len:], device=self._device)
                lls.append(sl.gather(-1, tgt[:, None]).squeeze(-1).sum())   # 0-dim GPU tensors
                greedys.append((sl.argmax(-1) == tgt).all())
                idxs.append(i)
            if idxs:
                ll_list = torch.stack(lls).tolist()              # single sync
                gd_list = torch.stack(greedys).tolist()          # single sync
                for i, ll, g in zip(idxs, ll_list, gd_list):
                    results[i] = (float(ll), bool(g))
        return results

    # ---- loglikelihood_rolling: total logprob of a full string (non-overlapping windows) ----
    @torch.no_grad()
    def loglikelihood_rolling(self, requests) -> list[float]:
        out: list[float] = []
        for r in requests:
            toks = self.tok_encode(r.args[0])
            if not toks:
                out.append(0.0)
                continue
            inp_full = [self.eot] + toks                          # inp_full[k] predicts toks[k]
            total = 0.0
            for s in range(0, len(toks), self._max_length):
                tgt = toks[s:s + self._max_length]
                inp = inp_full[s:s + len(tgt)]
                logp = self._logp_slice(self._logits([inp])[0])   # (len(tgt), vocab)
                t = torch.tensor(tgt, device=self._device)
                total += float(logp[torch.arange(len(tgt), device=self._device), t].sum().item())
            out.append(total)
        return out

    # ---- generate_until: greedy decode with stop strings (slow; no KV cache) ----
    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        out: list[str] = []
        for r in requests:
            ctx, gen_kwargs = r.args
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_gen = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            ids = self.tok_encode(ctx)[-self._max_length:]
            idx = torch.tensor([ids], dtype=torch.long, device=self._device)
            gen: list[int] = []
            stopped = False
            for _ in range(max_gen):
                with self.amp:
                    logits, _ = self.model(idx[:, -self._max_length:])
                logits = logits[:, -1, :].float()
                logits[:, GPT2_VALID:] = -float("inf")
                nxt = int(logits.argmax(-1).item())
                if nxt == self.eot:
                    break
                gen.append(nxt)
                idx = torch.cat([idx, torch.tensor([[nxt]], device=self._device)], dim=1)
                text = self.tok_decode(gen)
                hits = [text.find(stop) for stop in until if stop and stop in text]
                if hits:
                    out.append(text[:min(hits)])
                    stopped = True
                    break
            if not stopped:
                out.append(self.tok_decode(gen))
        return out
