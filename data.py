"""FineWeb .bin data loader (modded-nanogpt format).

Each shard: 256-int32 header (magic 20240520, version 1, num_tokens), then uint16 GPT-2 tokens.
Yields (inputs, targets) of shape (B, S), next-token aligned per row. Shards by DDP rank.
"""
from __future__ import annotations

import glob
import torch


def load_shard(path: str) -> torch.Tensor:
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0].item() == 20240520, f"bad magic in {path}"
    assert header[1].item() == 1, f"bad version in {path}"
    ntok = int(header[2].item())
    tokens = torch.empty(ntok, dtype=torch.uint16)
    with open(path, "rb", buffering=0) as f:
        f.seek(256 * 4)
        nread = f.readinto(tokens.numpy())
    assert nread == ntok * 2, f"short read in {path}"
    return tokens


def data_generator(pattern: str, B: int, S: int, device, rank: int = 0, world: int = 1):
    files = sorted(glob.glob(pattern))
    assert files, f"no data files match {pattern}"
    block = B * S
    while True:
        for fp in files:
            toks = load_shard(fp)
            n_blocks = (len(toks) - 1) // block
            # interleave blocks across ranks so each rank sees distinct data
            for i in range(rank, n_blocks, world):
                buf = toks[i * block: i * block + block + 1].to(device, dtype=torch.long, non_blocking=True)
                x = buf[:-1].view(B, S)
                y = buf[1:].view(B, S)
                yield x, y


def count_tokens(pattern: str) -> int:
    total = 0
    for fp in sorted(glob.glob(pattern)):
        header = torch.from_file(str(fp), False, 256, dtype=torch.int32)
        total += int(header[2].item())
    return total
