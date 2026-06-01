"""Audio-caption dataset for stage-1 audio alignment (Clotho via HF datasets, parquet with embedded audio).

Mirrors the vision pretrain format: logical = [AUDIO_TOKEN] + caption + [EOT]; input=logical[:-1],
target=logical[1:]. MoEVLM expands AUDIO_TOKEN into CLAP features and carries the post-audio target onto
the last feature. Returns the raw waveform (CLAP's processor turns it into mel features in the train loop).
"""
from __future__ import annotations

import tiktoken
import torch
from torch.utils.data import Dataset

from multimodal import AUDIO_TOKEN, IGNORE_INDEX

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class ClothoAudio(Dataset):
    def __init__(self, repo: str = "CLAPv2/Clotho", split: str | None = None, sr: int = 48000, max_cap: int = 64):
        from datasets import load_dataset, Audio
        dd = load_dataset(repo)
        if split is None:
            split = "train" if "train" in dd else list(dd.keys())[0]
        ds = dd[split]
        cols = ds.column_names
        self.audio_col = next(c for c in cols if "audio" in c.lower())
        self.cap_col = next(c for c in cols if "cap" in c.lower() or "text" in c.lower())
        ds = ds.cast_column(self.audio_col, Audio(sampling_rate=sr))
        self.ds = ds
        self.sr = sr
        self.max_cap = max_cap
        print(f"[ClothoAudio] {repo}:{split} n={len(ds)} audio_col={self.audio_col} cap_col={self.cap_col}", flush=True)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        wav = ex[self.audio_col]["array"]
        cap = ex[self.cap_col]
        if isinstance(cap, (list, tuple)):
            cap = cap[0] if cap else ""
        cap_ids = ENC.encode_ordinary(" " + str(cap).strip())[:self.max_cap] + [EOT]
        logical = [AUDIO_TOKEN] + cap_ids
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor(logical[1:], dtype=torch.long)
        return torch.as_tensor(wav, dtype=torch.float32), ids, tgt


def audio_collate(batch):
    """Returns (list of 1-D waveform tensors, padded input_ids, padded targets)."""
    wavs, ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
    return [w.numpy() for w in wavs], pad_ids, pad_tgt
