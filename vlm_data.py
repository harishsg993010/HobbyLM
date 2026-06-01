"""LLaVA-Pretrain (LAION-CC-SBU-558K) dataset for stage-1 alignment.

Each sample -> a logical sequence [IMAGE_TOKEN] + caption_tokens + [EOT]; we train next-token, so
input_ids = logical[:-1], targets = logical[1:]. MoEVLM expands the IMAGE_TOKEN into 729 SigLIP2
features and carries the post-image target onto the last feature (see multimodal.build_inputs_embeds),
so the model learns to produce the caption conditioned on the image.
"""
from __future__ import annotations

import io
import json
import zipfile

import tiktoken
import torch
from PIL import Image
from torch.utils.data import Dataset

from multimodal import IMAGE_TOKEN, VIDEO_TOKEN, IGNORE_INDEX

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


class LlavaPretrain(Dataset):
    """Streams images directly from images.zip by name (no extraction). Each DataLoader worker opens
    its own ZipFile handle lazily (ZipFile isn't fork/thread-safe to share)."""
    def __init__(self, json_path: str, zip_path: str, max_cap: int = 128):
        with open(json_path) as f:
            self.data = json.load(f)
        self.zip_path = zip_path
        self.max_cap = max_cap
        self._zip = None

    def _z(self) -> zipfile.ZipFile:
        if self._zip is None:                       # opened per-worker after fork
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        # conversations: [{"from":"human","value":"<image>\n..."}, {"from":"gpt","value":"<caption>"}]
        caption = ex["conversations"][1]["value"].strip()
        cap_ids = ENC.encode_ordinary(" " + caption)[:self.max_cap] + [EOT]
        logical = [IMAGE_TOKEN] + cap_ids
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor(logical[1:], dtype=torch.long)
        try:
            img = Image.open(io.BytesIO(self._z().read(ex["image"]))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384))   # tolerate a missing/corrupt image
        return img, ids, tgt


class LlavaSFT(Dataset):
    """LLaVA-Instruct-150K instruction tuning. Builds a multi-turn sequence
    [IMAGE] USER: q1 ASSISTANT: a1<eot> USER: q2 ASSISTANT: a2<eot> ... and trains next-token loss
    ONLY on the assistant answers (+ their EOT); image/scaffolding/user tokens are IGNORE.
    Images streamed from train2017.zip as 'train2017/<image>'."""
    def __init__(self, json_path: str, zip_path: str, max_len: int = 1024, sentinel: int = IMAGE_TOKEN):
        with open(json_path) as f:
            self.data = json.load(f)
        self.zip_path = zip_path
        self.max_len = max_len
        self.sentinel = sentinel          # IMAGE_TOKEN, or VIDEO_TOKEN to present the same data as video
        self._zip = None

    def _z(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        # logical token stream + a per-token "is assistant answer" mask
        logical = [self.sentinel]                  # IMAGE_TOKEN or VIDEO_TOKEN
        mask = [False]
        for turn in ex["conversations"]:
            val = turn["value"].replace("<image>", "").strip()
            if turn["from"] == "human":
                t = ENC.encode_ordinary("USER: " + val + "\nASSISTANT:")
                logical += t
                mask += [False] * len(t)
            else:  # gpt / assistant
                t = ENC.encode_ordinary(" " + val) + [EOT]
                logical += t
                mask += [True] * len(t)            # train on the answer (+ EOT)
        logical, mask = logical[:self.max_len], mask[:self.max_len]
        # next-token: input=logical[:-1], target=logical[1:] but only where the TARGET is an answer token
        ids = torch.tensor(logical[:-1], dtype=torch.long)
        tgt = torch.tensor([logical[k + 1] if mask[k + 1] else IGNORE_INDEX
                            for k in range(len(logical) - 1)], dtype=torch.long)
        try:
            img = Image.open(io.BytesIO(self._z().read("train2017/" + ex["image"]))).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384))
        return img, ids, tgt


def collate(batch):
    """Pad input_ids (with EOT) and targets (with IGNORE) to the batch max; return (PIL list, ids, tgt)."""
    imgs, ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
    return list(imgs), pad_ids, pad_tgt
