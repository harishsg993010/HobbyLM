"""Frozen Whisper speech encoder for the MoE-VLM (speech-understanding path, distinct from CLAP audio).

CLAP captions *what kind of sound* a clip is; Whisper carries *the spoken words*. We load the encoder
half of `openai/whisper-small` (the decoder is discarded), run 16 kHz waveforms through it under no_grad,
and return a token sequence (B, T, hidden) to be projected + spliced at <speech> sentinels by MoEVLM —
the same mechanism as image patches / CLAP frames, just a different encoder + sentinel + projector.

Whisper always pads/truncates to 30 s -> 1500 encoder frames @ 50 Hz. We STACK `stack` adjacent frames
(concat along channels) to cut the sequence length (1500 -> 1500/stack tokens) while preserving detail,
which is what speech-LLMs (Ultravox/Qwen-Audio) do; hidden becomes d_model*stack. Frozen in every stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn

WHISPER_ID = "openai/whisper-small"
WHISPER_SR = 16000


class WhisperSpeech(nn.Module):
    def __init__(self, model_id: str = WHISPER_ID, device="cuda", dtype=torch.bfloat16, stack: int = 2):
        super().__init__()
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(model_id)
        full = WhisperModel.from_pretrained(model_id, torch_dtype=dtype)
        self.encoder = full.encoder.to(device).eval()          # discard the decoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.device = device
        self.dtype = dtype
        self.stack = stack
        self.d = full.config.d_model                           # 768 for whisper-small
        self.hidden = self.d * stack                           # projector input dim

    @torch.no_grad()
    def encode(self, waveforms, sr: int = WHISPER_SR) -> torch.Tensor:
        """waveforms: list of 1-D float arrays (mono, 16 kHz). Returns (B, 1500/stack, d*stack)."""
        feats = self.fe(waveforms, sampling_rate=sr, return_tensors="pt").input_features  # (B, 80, 3000)
        feats = feats.to(self.device, self.dtype)
        h = self.encoder(feats).last_hidden_state              # (B, 1500, d)
        B, T, C = h.shape
        s = self.stack
        if T % s:                                              # pad so T divides by stack
            pad = s - (T % s)
            h = torch.cat([h, h[:, -1:].expand(B, pad, C)], dim=1)
            T += pad
        return h.reshape(B, T // s, C * s)                     # stack adjacent frames along channels
