"""Multimodal (image + audio) wrapper around the MoE LLM — TinyLLaVA-style.

A frozen vision/audio encoder produces patch features; a small MLP projector maps them into the
LLM's embedding space; the LLM consumes them as ordinary token embeddings spliced at `<image>` /
`<audio>` sentinel positions. The LLM is unchanged except `forward(inputs_embeds=...)`.

Sentinels live in the padded-but-unused GPT-2 vocab slots (50257-50303), so they never collide with
real tokens and the lm_head already masks them at decode (see generate.py).

v1 scope: encoders frozen (features precomputed/cached); projector(s) + (optionally) LLM are trained.
Batching assumes uniform feature count per modality (fixed-resolution encoders); samples may carry an
image, audio, both, or neither (text-only) — the merge right-pads, which is safe under causal attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .config import ModelConfig
from .model import MoETransformer

# ---- special tokens in the free vocab slots (vocab 50257 real -> padded 50304) ----
IMAGE_TOKEN = 50257   # one sentinel per image; replaced by N projected patch features
AUDIO_TOKEN = 50258   # one sentinel per audio clip
IM_START = 50259
IM_END = 50260
VIDEO_TOKEN = 50261   # video = sampled frames through the SAME vision encoder + mm_projector (no new encoder)
SPEECH_TOKEN = 50262  # spoken language via a Whisper encoder + speech_projector (distinct from CLAP <audio>)
IGNORE_INDEX = -1     # target value for non-text positions; matches model CE ignore_index


class Projector(nn.Module):
    """TinyLLaVA `mlp2x_gelu` connector: Linear -> GELU -> Linear, encoder_dim -> d_model."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MoEVLM(nn.Module):
    """Wraps a MoETransformer with image (and optional audio) projectors.

    forward inputs:
      input_ids:       (B, L) token ids containing IMAGE_TOKEN / AUDIO_TOKEN sentinels.
      image_features:  (B, Ni, vision_dim) RAW frozen-encoder features, or None.
      audio_features:  (B, Na, audio_dim) RAW frozen-encoder features, or None.
      targets:         (B, L) next-token targets aligned to input_ids (sentinels get IGNORE_INDEX), or None.
    """
    def __init__(self, llm: MoETransformer, vision_dim: int = 1152, audio_dim: int | None = None,
                 speech_dim: int | None = None):
        super().__init__()
        self.llm = llm
        self.d_model = llm.cfg.d_model
        self.mm_projector = Projector(vision_dim, self.d_model)
        self.audio_projector = Projector(audio_dim, self.d_model) if audio_dim else None
        self.speech_projector = Projector(speech_dim, self.d_model) if speech_dim else None

    # ---- build the merged (B, L', d) embedding sequence by splicing modality features ----
    def build_inputs_embeds(self, input_ids: Tensor, image_features: Tensor | None = None,
                            audio_features: Tensor | None = None, targets: Tensor | None = None,
                            video_features: Tensor | None = None, speech_features: Tensor | None = None):
        B, L = input_ids.shape
        dev = input_ids.device
        img_proj = self.mm_projector(image_features) if image_features is not None else None  # (B,Ni,d)
        vid_proj = self.mm_projector(video_features) if video_features is not None else None   # video reuses mm_projector
        aud_proj = (self.audio_projector(audio_features)
                    if (audio_features is not None and self.audio_projector is not None) else None)
        spk_proj = (self.speech_projector(speech_features)
                    if (speech_features is not None and self.speech_projector is not None) else None)

        seqs_e, seqs_t = [], []
        for b in range(B):
            ids = input_ids[b]
            text_emb = self.llm.embed(ids)                              # (L, d); sentinel rows get sliced out
            # ordered list of (position, feature_block) for every sentinel in this sample
            spots = []
            if img_proj is not None:
                for p in (ids == IMAGE_TOKEN).nonzero(as_tuple=True)[0]:
                    spots.append((int(p), img_proj[b]))
            if vid_proj is not None:
                for p in (ids == VIDEO_TOKEN).nonzero(as_tuple=True)[0]:
                    spots.append((int(p), vid_proj[b]))
            if aud_proj is not None:
                for p in (ids == AUDIO_TOKEN).nonzero(as_tuple=True)[0]:
                    spots.append((int(p), aud_proj[b]))
            if spk_proj is not None:
                for p in (ids == SPEECH_TOKEN).nonzero(as_tuple=True)[0]:
                    spots.append((int(p), spk_proj[b]))
            spots.sort(key=lambda s: s[0])

            e_parts, t_parts, prev = [], [], 0
            for pos, feat in spots:
                e_parts.append(text_emb[prev:pos])
                e_parts.append(feat.to(text_emb.dtype))   # encoder/projector may be bf16 under autocast
                if targets is not None:
                    t_parts.append(targets[b][prev:pos])
                    # next-token: the LAST feature predicts the token following the sentinel (no internal
                    # label shift in our model), so carry targets[pos] onto it; the rest are ignored.
                    ft = torch.full((feat.shape[0],), IGNORE_INDEX, dtype=targets.dtype, device=dev)
                    ft[-1] = targets[b][pos]
                    t_parts.append(ft)
                prev = pos + 1
            e_parts.append(text_emb[prev:])
            if targets is not None:
                t_parts.append(targets[b][prev:])
            seqs_e.append(torch.cat(e_parts, dim=0))
            if targets is not None:
                seqs_t.append(torch.cat(t_parts, dim=0))

        # right-pad to the longest merged sequence (causal attention -> pads don't affect real tokens)
        Lmax = max(e.shape[0] for e in seqs_e)
        inputs_embeds = seqs_e[0].new_zeros(B, Lmax, self.d_model)
        new_targets = None
        if targets is not None:
            new_targets = torch.full((B, Lmax), IGNORE_INDEX, dtype=targets.dtype, device=dev)
        for b, e in enumerate(seqs_e):
            inputs_embeds[b, :e.shape[0]] = e
            if targets is not None:
                new_targets[b, :seqs_t[b].shape[0]] = seqs_t[b]
        return inputs_embeds, new_targets

    def forward(self, input_ids: Tensor, image_features: Tensor | None = None,
                audio_features: Tensor | None = None, targets: Tensor | None = None,
                video_features: Tensor | None = None, speech_features: Tensor | None = None):
        inputs_embeds, new_targets = self.build_inputs_embeds(
            input_ids, image_features, audio_features, targets, video_features, speech_features)
        return self.llm(inputs_embeds=inputs_embeds, targets=new_targets)

    # ---- param groups: freeze encoders (external), optionally freeze the LLM (stage 1) ----
    def set_llm_trainable(self, trainable: bool):
        for p in self.llm.parameters():
            p.requires_grad = trainable

    def projector_parameters(self):
        ps = list(self.mm_projector.parameters())
        if self.audio_projector is not None:
            ps += list(self.audio_projector.parameters())
        if self.speech_projector is not None:
            ps += list(self.speech_projector.parameters())
        return ps


def build_vlm(cfg: ModelConfig, vision_dim: int = 1152, audio_dim: int | None = None) -> MoEVLM:
    return MoEVLM(MoETransformer(cfg), vision_dim=vision_dim, audio_dim=audio_dim)
