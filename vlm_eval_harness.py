"""lmms-eval (lmms-lab) harness wrapper for the MoE-VLM — the multimodal twin of
`eval_harness.MoELMWrapper`.

Registers a `moe_vlm` model: SigLIP2-encodes the doc image, splices it at IMAGE_TOKEN, wraps the
task's question in our trained USER:/ASSISTANT: chat format, and greedy-decodes the answer with the
SAME rep-penalty + no-repeat-3gram used in caption(). Runs POPE / GQA / VQAv2 (all `generate_until`
tasks) through `lmms_eval.evaluator.simple_evaluate`. See modal_mm.py `--action vlm_eval`.
"""
from __future__ import annotations

import torch
import tiktoken

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from vision import SiglipVision
from multimodal import MoEVLM, IMAGE_TOKEN
from generate import load_model, GPT2_VALID, EOT

BACKBONE = "/data/runs/500M_ctx2048/model.pt"


def _banned_ngram(prev, n=3):
    """Tokens that would complete a repeated n-gram given the generated prefix `prev`."""
    if len(prev) < n:
        return []
    seen = {}
    for j in range(len(prev) - n + 1):
        seen.setdefault(tuple(prev[j:j + n - 1]), []).append(prev[j + n - 1])
    return seen.get(tuple(prev[-(n - 1):]), [])


@register_model("moe_vlm")
class MoEVLMHarness(lmms):
    """`model_args` (k=v,...): stage2_run, joint_run (optional override), backbone, max_new, rep_pen."""

    def __init__(self, stage2_run: str = "500M_vlm_stage2", joint_run: str = "",
                 backbone: str = BACKBONE, max_new: int = 64, rep_pen: float = 1.3,
                 max_length: int = 2048, device: str = "cuda", vision_id: str = "", **kwargs):
        super().__init__()
        device = device or "cuda"                          # additional_config may pass device=None
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._rank, self._world_size = 0, 1                # single-process eval
        self.max_new = int(max_new)
        self.rep_pen = float(rep_pen)
        self._max_length = int(max_length)

        self.enc_img = SiglipVision(model_id=vision_id, device=self._device) if vision_id \
            else SiglipVision(device=self._device)
        llm, cfg, _, _ = load_model(backbone, self._device)
        self.vlm = MoEVLM(llm, vision_dim=self.enc_img.hidden).to(self._device)
        run = joint_run or stage2_run                      # joint ckpt is drop-in (same model/projector keys)
        ck = torch.load(f"/data/runs/{run}/model.pt", map_location=self._device, weights_only=False)
        self.vlm.llm.load_state_dict(ck["model"])
        self.vlm.mm_projector.load_state_dict(ck["projector"])
        self.vlm.eval()
        self.tok = tiktoken.get_encoding("gpt2")
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16)
        print(f"[moe_vlm] loaded /data/runs/{run}/model.pt | d{cfg.d_model} L{cfg.n_layers} "
              f"| max_new={self.max_new} rep_pen={self.rep_pen}", flush=True)

    # ---- properties the harness reads ----
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @staticmethod
    def _flatten_visuals(visuals):
        out = []
        stack = list(visuals) if isinstance(visuals, (list, tuple)) else [visuals]
        while stack:
            v = stack.pop(0)
            if isinstance(v, (list, tuple)):
                stack = list(v) + stack
            elif v is not None:
                out.append(v)
        return out

    @torch.no_grad()
    def _answer(self, image, question: str, until, max_new: int) -> str:
        """One greedy answer for (image, question), in the trained chat format. Stops on EOT or any
        `until` string (lmms-eval passes these per task), else after max_new tokens."""
        q = question.replace("<image>", "").strip()
        pre = self.tok.encode_ordinary(f"USER: {q}\nASSISTANT:")
        if image is not None:
            feats = self.enc_img.encode([image])
            ids = torch.tensor([[IMAGE_TOKEN] + pre], device=self._device)
            with self.amp:
                cur, _ = self.vlm.build_inputs_embeds(ids, image_features=feats)
        else:                                              # text-only fallback (rare)
            ids = torch.tensor([pre], device=self._device)
            with self.amp:
                cur = self.vlm.llm.embed(ids)
        outs: list[int] = []
        for _ in range(max_new):
            with self.amp:
                lg = self.vlm.llm(inputs_embeds=cur)[0][:, -1, :].float()
            lg[:, GPT2_VALID:] = -float("inf")
            if outs:                                       # CTRL-style repetition penalty
                u = torch.tensor(sorted(set(outs)), device=self._device)
                v = lg[0, u]
                lg[0, u] = torch.where(v > 0, v / self.rep_pen, v * self.rep_pen)
            for b in _banned_ngram(outs, 3):
                lg[0, b] = -float("inf")
            t = int(lg.argmax(-1).item())
            if t == EOT:
                break
            outs.append(t)
            text = self.tok.decode(outs)
            hit = [text.find(s) for s in until if s and s in text]
            if hit:
                return text[:min(hit)]
            cur = torch.cat([cur, self.vlm.llm.embed(torch.tensor([[t]], device=self._device)).to(cur.dtype)], dim=1)
        return self.tok.decode(outs)

    # function words to ignore when reducing a sentence answer to its salient token
    _STOP = set((
        "a an the is are was were be been being am of on in at to and or with this that these those "
        "it its they them their there here over under near by from as into onto for has have had does "
        "do did then than so but not no yes i we you he she his her him my your our what which who "
        "whom whose where when why how many much kind type look looks looking wearing using single "
        "word phrase question answer image photo picture appears seems showing shows contains located"
    ).split())

    @classmethod
    def _short_answer(cls, question: str, ans: str) -> str:
        """Reduce a verbose sentence answer to the concise token the metric scores. Leading yes/no wins
        (most GQA binary Qs); otherwise pick the last content word that is NOT already in the question
        (the genuinely new information), e.g. 'A man is wearing the dress.' -> 'man'."""
        low = ans.strip().lower()
        head = low.replace(".", " ").replace(",", " ").split()
        if head and head[0] in ("yes", "no"):
            return head[0]
        import re
        q = question.lower().split("answer the question")[0]
        qwords = set(re.findall(r"[a-z']+", q))
        awords = re.findall(r"[a-z']+", low)
        content = [w for w in awords if w not in cls._STOP and w not in qwords]
        return content[-1] if content else (awords[-1] if awords else low)

    def _normalize(self, task: str, question: str, ans: str) -> str:
        """Coerce our caption-tuned output into the form each metric scores. POPE = exact-match yes/no;
        GQA/VQAv2 = short-answer exact/VQA-accuracy. Without this a correct-but-verbose answer scores 0."""
        if task.startswith("pope"):
            low = ans.strip().lower().replace(".", " ").replace(",", " ").split()
            if low and low[0] in ("yes", "no"):
                return low[0]
            return "yes" if "yes" in low[:3] else ("no" if "no" in low[:3] else " ".join(low))
        if task.startswith("gqa") or task.startswith("vqav2"):
            return self._short_answer(question, ans)
        return ans.strip()

    # ---- generate_until: the path POPE / GQA / VQAv2 use ----
    @torch.no_grad()
    def generate_until(self, requests) -> list[str]:
        res = []
        for n, req in enumerate(requests):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = req.arguments
            visuals = self._flatten_visuals(doc_to_visual(self.task_dict[task][split][doc_id]))
            until = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            # short-answer tasks don't need 64 tokens; cap tight for speed (pope=yes/no, gqa/vqa=a phrase)
            cap = 8 if task.startswith("pope") else 32
            max_new = min(cap, int(gen_kwargs.get("max_new_tokens", gen_kwargs.get("max_gen_toks", self.max_new))))
            img = visuals[0] if visuals else None
            raw = self._answer(img, contexts, until, max_new).strip()
            ans = self._normalize(task, contexts, raw)
            res.append(ans)
            if n < 6:
                print(f"[moe_vlm] {task} #{n} Q={contexts[:60]!r} RAW={raw!r} -> {ans!r}", flush=True)
            elif n % 500 == 0:
                print(f"[moe_vlm] {task}: {n}/{len(requests)}", flush=True)
        return res

    # ---- loglikelihood: implemented for completeness (POPE/GQA/VQAv2 don't use it) ----
    @torch.no_grad()
    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        import torch.nn.functional as F
        out = []
        for req in requests:
            contexts, target, doc_to_visual, doc_id, task, split = req.arguments
            visuals = self._flatten_visuals(doc_to_visual(self.task_dict[task][split][doc_id]))
            img = visuals[0] if visuals else None
            q = contexts.replace("<image>", "").strip()
            pre = self.tok.encode_ordinary(f"USER: {q}\nASSISTANT:")
            cont = self.tok.encode_ordinary(" " + target.strip())
            ids = torch.tensor([([IMAGE_TOKEN] if img is not None else []) + pre + cont], device=self._device)
            feats = self.enc_img.encode([img]) if img is not None else None
            with self.amp:
                emb, _ = self.vlm.build_inputs_embeds(ids, image_features=feats)
                logits = self.vlm.llm(inputs_embeds=emb)[0].float()
            logits[..., GPT2_VALID:] = -float("inf")
            logp = F.log_softmax(logits, dim=-1)[0]
            # the last len(cont) positions of `emb` predict cont (image expands but stays left of cont)
            tgt = torch.tensor(cont, device=self._device)
            sl = logp[-len(cont) - 1:-1, :]
            ll = sl.gather(-1, tgt[:, None]).squeeze(-1).sum().item()
            greedy = bool((sl.argmax(-1) == tgt).all().item())
            out.append((float(ll), greedy))
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("moe_vlm: rolling loglikelihood not needed for POPE/GQA/VQAv2")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("moe_vlm: multi-round generation not used by POPE/GQA/VQAv2")
