"""Frozen SigLIP2 vision encoder wrapper for the MoE-VLM.

Loads `google/siglip2-so400m-patch14-384` (or any SigLIP/SigLIP2), runs images through its vision
tower under no_grad, and returns patch features (B, N, hidden) to be projected + spliced by MoEVLM.
The encoder is frozen in every training stage, so we run it on the fly (precomputing features for
558K images would be ~900 GB). Lazy transformers import so the module is CPU-importable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SIGLIP2_ID = "google/siglip2-so400m-patch14-384"


class SiglipVision(nn.Module):
    def __init__(self, model_id: str = SIGLIP2_ID, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        # AutoImageProcessor, NOT AutoProcessor: we only do image preprocessing here. AutoProcessor also
        # tries to load the SigLIP2 text tokenizer (SentencePiece), which fails on some transformers
        # builds (vocab_file=None) — and we never use it.
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        full = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.vision = full.vision_model.to(device).eval()
        for p in self.vision.parameters():
            p.requires_grad = False
        self.device = device
        self.dtype = dtype
        self.hidden = self.vision.config.hidden_size

    @torch.no_grad()
    def preprocess(self, images) -> torch.Tensor:
        """images: list of PIL.Image -> pixel_values (B, 3, H, W) on device."""
        px = self.processor(images=images, return_tensors="pt").pixel_values
        return px.to(self.device, self.dtype)

    @torch.no_grad()
    def encode_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (B,3,H,W) -> patch features (B, N, hidden)."""
        return self.vision(pixel_values=pixel_values).last_hidden_state

    @torch.no_grad()
    def encode(self, images) -> torch.Tensor:
        """list of PIL.Image -> (B, N, hidden) patch features."""
        return self.encode_pixels(self.preprocess(images))
