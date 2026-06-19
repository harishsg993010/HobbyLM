"""Streaming datasets for fine-tuning on the big NVIDIA sets WITHOUT downloading them.

- StreamAgentic: streams nvidia/Nemotron-SFT-Agentic-v2 jsonl (interactive_agent/search/tool_calling)
  line-by-line from HF, reuses tool_data.extract_trajectory -> multi-turn (ids, targets).
- StreamVLMCaption: streams the VLM captioning jsonl and fetches each image from OpenImages S3 by URL,
  ShareGPT conversations -> LLaVA-style (PIL image, ids, targets).
- StreamVLMOcr: streams an OCR jsonl + its WebDataset tar shards (filename->bytes) for the images.

All are IterableDatasets sharded by (rank, worker) so DDP + workers each see a disjoint slice; they loop
forever (the joint trainer pulls a fixed number of steps).
"""
from __future__ import annotations

import io
import json
import tarfile

import tiktoken
import torch
from torch.utils.data import IterableDataset, get_worker_info

from multimodal import IMAGE_TOKEN, IGNORE_INDEX
from tool_data import extract_trajectory

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256
OPENIMAGES = "https://s3.amazonaws.com/open-images-dataset/train/{}"


def _shard(rank, world):
    wi = get_worker_info()
    nw = wi.num_workers if wi else 1
    wid = wi.id if wi else 0
    return rank * nw + wid, world * nw


def _local(repo, path):
    """Resolve a (jsonl) file to a local path. Prefer the cache (local_files_only -> NO network HEAD, so
    the many dataloader workers don't storm HF with concurrent SSL connections); the modal launcher
    pre-downloads these once before torchrun. Falls back to a real download if not cached."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, path, repo_type="dataset", local_files_only=True)
    except Exception:
        return hf_hub_download(repo, path, repo_type="dataset")


def _file_lines(local_paths, shard, nshards):
    """Infinite, sharded line stream over local jsonl files."""
    idx = 0
    while True:
        for p in local_paths:
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if idx % nshards == shard:
                            yield line
                        idx += 1
            except Exception:
                continue


def _conv_to_ids(conversations, image_token=IMAGE_TOKEN, max_len=2048):
    """ShareGPT [{from,value}] with <image> -> (ids, targets); loss on gpt answers. One <image> -> sentinel."""
    ids, loss = [image_token], [0]                      # leading image sentinel
    for m in conversations:
        val = (m.get("value") or "").replace("<image>", "").strip()
        if m.get("from") == "human":
            t = ENC.encode_ordinary("USER: " + val + "\nASSISTANT:")
            ids += t; loss += [0] * len(t)
        elif m.get("from") == "gpt":
            t = ENC.encode_ordinary(" " + val) + [EOT]
            ids += t; loss += [1] * len(t)
    ids, loss = ids[:max_len], loss[:max_len]
    inp = torch.tensor(ids[:-1], dtype=torch.long)
    tgt = torch.tensor([ids[k + 1] if loss[k + 1] else IGNORE_INDEX for k in range(len(ids) - 1)], dtype=torch.long)
    return inp, tgt


class StreamAgentic(IterableDataset):
    def __init__(self, repo="nvidia/Nemotron-SFT-Agentic-v2",
                 files=("data/tool_calling.jsonl", "data/search.jsonl", "data/interactive_agent.jsonl"),
                 rank=0, world=1, max_len=2048):
        self.repo, self.files = repo, files
        self.rank, self.world, self.max_len = rank, world, max_len

    def __iter__(self):
        shard, n = _shard(self.rank, self.world)
        paths = [_local(self.repo, f) for f in self.files]
        for line in _file_lines(paths, shard, n):
            try:
                segs = extract_trajectory(json.loads(line), "nemotron")
            except Exception:
                segs = None
            if not segs:
                continue
            yield _segs_to_ids(segs, self.max_len)


def _segs_to_ids(segs, max_len):
    ids, loss = [], []
    for text, is_loss in segs:
        t = ENC.encode_ordinary(text)
        if is_loss:
            t = t + [EOT]
        ids += t; loss += [1 if is_loss else 0] * len(t)
    ids, loss = ids[:max_len], loss[:max_len]
    inp = torch.tensor(ids[:-1], dtype=torch.long)
    tgt = torch.tensor([ids[k + 1] if loss[k + 1] else IGNORE_INDEX for k in range(len(ids) - 1)], dtype=torch.long)
    return inp, tgt


class StreamVLMCaption(IterableDataset):
    """Captioning jsonl + OpenImages S3 images (streamed by URL)."""
    def __init__(self, repo="nvidia/Llama-Nemotron-VLM-Dataset-v1",
                 files=("captioning_1.jsonl", "captioning_2.jsonl"), rank=0, world=1, max_len=2048):
        self.repo, self.files = repo, files
        self.rank, self.world, self.max_len = rank, world, max_len

    def __iter__(self):
        import requests
        from PIL import Image
        shard, n = _shard(self.rank, self.world)
        paths = [_local(self.repo, f) for f in self.files]
        for line in _file_lines(paths, shard, n):
            try:
                ex = json.loads(line)
                r = requests.get(OPENIMAGES.format(ex["image"]), timeout=12)
                if r.status_code != 200:
                    continue
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                inp, tgt = _conv_to_ids(ex["conversations"], max_len=self.max_len)
            except Exception:
                continue
            yield img, inp, tgt


class StreamOcr(IterableDataset):
    """OCR from Llama-Nemotron-VLM-v1 ocr_4: rendered English-Wikipedia text images (read text -> markdown/LaTeX).
    The jsonl (image->conversations) and the WebDataset tar shards are PRE-STAGED to the HF cache by the launcher
    (HF_HUB_DISABLE_XET=1, single process); here every file is resolved LOCALLY via _local -> NO Xet/SSL at train
    time. Tars are sharded across (rank, worker); each image in a tar is paired to its conversation by basename."""
    def __init__(self, repo="nvidia/Llama-Nemotron-VLM-Dataset-v1", part="ocr_4", n_shards=9,
                 rank=0, world=1, max_len=2048, max_bytes=8_000_000, max_px=2048):
        self.repo, self.part, self.n_shards = repo, part, n_shards
        self.rank, self.world, self.max_len = rank, world, max_len
        self.max_bytes, self.max_px = max_bytes, max_px       # hang guards: skip >8MB members, cap dims

    def __iter__(self):
        from PIL import Image
        shard, n = _shard(self.rank, self.world)
        # basename(no-ext) -> conversations, from the local jsonl (built once per worker)
        meta = {}
        try:
            with open(_local(self.repo, f"{self.part}.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    meta[ex["image"].rsplit("/", 1)[-1].rsplit(".", 1)[0]] = ex["conversations"]
        except Exception:
            return
        tars = [f"{self.part}_images/shard_{i:06d}.tar" for i in range(self.n_shards)]
        mine = tars[shard::n] or tars            # 9 tars < workers -> idle workers fall back to the full list
        while True:
            for tp in mine:
                try:
                    local_tar = _local(self.repo, tp)
                except Exception:
                    continue
                try:
                    with tarfile.open(local_tar) as tf:
                        for m in tf:
                            if not m.isfile():
                                continue
                            # HANG GUARD: skip anomalously large members — a giant image decode can block a
                            # dataloader worker, which starves next() on one rank and DDP-deadlocks the all-reduce
                            # (the c10d timeout that killed joint10). OCR doc PNGs are well under this.
                            if m.size and m.size > self.max_bytes:
                                continue
                            base = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                            conv = meta.get(base)
                            if conv is None:
                                continue
                            try:
                                ef = tf.extractfile(m)
                                if ef is None:
                                    continue
                                img = Image.open(io.BytesIO(ef.read())).convert("RGB")
                                if max(img.size) > self.max_px:        # cap dimensions before the slow path
                                    img.thumbnail((self.max_px, self.max_px))
                                inp, tgt = _conv_to_ids(conv, max_len=self.max_len)
                            except Exception:
                                continue
                            yield img, inp, tgt
                except Exception:
                    continue


class StreamVLM(IterableDataset):
    """VLM image stream. Captioning images come from OpenImages S3 (no HF Xet), so this is the reliable
    streaming path. (OCR images live in HF Xet tar shards which stream unreliably from Modal -> StreamVLMOcr
    is available but not used here.)"""
    def __init__(self, rank=0, world=1, max_len=2048):
        self.cap = StreamVLMCaption(rank=rank, world=world, max_len=max_len)

    def __iter__(self):
        yield from self.cap


class NonBlockingLoader:
    """DDP hang guard for streaming paths. A background thread pulls from the underlying (infinite) DataLoader
    into a bounded queue; __next__ pops with a timeout and, if the stream stalls, returns the LAST good batch
    so this rank still reaches the all-reduce. Without this, one stalled dataloader worker blocks next() forever
    -> the other ranks time out at the collective -> NCCL ChildFailedError (the joint10/joint11 hang).
    A repeated batch on a rare stall is harmless; a deadlocked rank is fatal."""
    def __init__(self, loader, timeout=45.0, maxsize=4):
        import threading, queue
        self._q = queue.Queue(maxsize=maxsize)
        self._Empty = queue.Empty
        self._last = None
        self._timeout = timeout
        self._it = iter(loader)
        self._t = threading.Thread(target=self._fill, daemon=True)
        self._t.start()

    def _fill(self):
        while True:
            try:
                b = next(self._it)            # underlying stream is infinite (no StopIteration)
            except Exception:
                continue
            self._q.put(b)                    # blocks if consumer is behind (back-pressure) — fine

    def __iter__(self):
        return self

    def __next__(self):
        if self._last is None:                # must block for the very first batch (training needs one)
            self._last = self._q.get()
            return self._last
        try:
            self._last = self._q.get(timeout=self._timeout)
        except self._Empty:
            pass                              # stream stalled -> reuse last good batch, keep DDP in lockstep
        return self._last


def _pack(ids, loss, max_len):
    if not any(loss):
        return None
    ids, loss = ids[:max_len], loss[:max_len]
    if len(ids) < 2:
        return None
    inp = torch.tensor(ids[:-1], dtype=torch.long)
    tgt = torch.tensor([ids[k + 1] if loss[k + 1] else IGNORE_INDEX for k in range(len(ids) - 1)], dtype=torch.long)
    return inp, tgt


def _msgs_to_ids(messages, max_len=2048):
    """smoltalk-style chat [{role,content}] -> (ids, targets); loss on assistant turns."""
    ids, loss = [], []
    for m in messages:
        role = m.get("role"); content = (m.get("content") or "").strip()
        if not content:
            continue
        if role in ("system", "developer"):
            t = ENC.encode_ordinary(content + "\n"); ids += t; loss += [0] * len(t)
        elif role == "user":
            t = ENC.encode_ordinary("USER: " + content + "\nASSISTANT:"); ids += t; loss += [0] * len(t)
        elif role == "assistant":
            t = ENC.encode_ordinary(" " + content) + [EOT]; ids += t; loss += [1] * len(t)
    return _pack(ids, loss, max_len)


class StreamSmolTalk(IterableDataset):
    """HuggingFaceTB/smoltalk 'all' config: multi-turn chat -> text ids (loss on assistant). Streams the
    pre-staged parquet shards by row group, sharded across (rank, worker)."""
    def __init__(self, repo="HuggingFaceTB/smoltalk",
                 files=tuple(f"data/all/train-{i:05d}-of-00009.parquet" for i in range(9)),
                 rank=0, world=1, max_len=2048):
        self.repo, self.files = repo, files
        self.rank, self.world, self.max_len = rank, world, max_len

    def __iter__(self):
        import pyarrow.parquet as pq
        shard, n = _shard(self.rank, self.world)
        paths = [_local(self.repo, f) for f in self.files]
        idx = 0
        while True:
            for p in paths:
                try:
                    pf = pq.ParquetFile(p)
                except Exception:
                    continue
                for rg in range(pf.num_row_groups):
                    try:
                        rows = pf.read_row_group(rg, columns=["messages"]).column("messages").to_pylist()
                    except Exception:
                        continue
                    for msgs in rows:
                        if idx % n == shard:
                            r = _msgs_to_ids(msgs, self.max_len)
                            if r is not None:
                                yield r
                        idx += 1


def _mobile_to_ids(ex, max_len=2048):
    """google/mobile-actions: tools + user query + assistant tool_calls -> (ids, targets); loss on the call."""
    tools = ex.get("tools") or []
    tools_str = json.dumps([t.get("function", t) for t in tools], separators=(",", ":"))
    ids, loss = ENC.encode_ordinary("TOOLS: " + tools_str + "\n"), None
    loss = [0] * len(ids)
    for m in ex.get("messages", []):
        role = m.get("role")
        if role == "user":
            t = ENC.encode_ordinary("USER: " + (m.get("content") or "").strip() + "\nASSISTANT:")
            ids += t; loss += [0] * len(t)
        elif role == "assistant":
            tc = m.get("tool_calls")
            if tc:
                calls = [{"name": (c.get("function") or {}).get("name"),
                          "arguments": (c.get("function") or {}).get("arguments", {})} for c in tc]
                t = ENC.encode_ordinary(" " + json.dumps(calls, separators=(",", ":"))) + [EOT]
            else:
                t = ENC.encode_ordinary(" " + (m.get("content") or "").strip()) + [EOT]
            ids += t; loss += [1] * len(t)
    return _pack(ids, loss, max_len)


class StreamMobileActions(IterableDataset):
    """google/mobile-actions: streamed mobile tool-calling (turn_off_flashlight / create_calendar_event / ...)."""
    def __init__(self, repo="google/mobile-actions", path="dataset.jsonl", rank=0, world=1, max_len=2048):
        self.repo, self.path = repo, path
        self.rank, self.world, self.max_len = rank, world, max_len

    def __iter__(self):
        shard, n = _shard(self.rank, self.world)
        for line in _file_lines([_local(self.repo, self.path)], shard, n):
            try:
                r = _mobile_to_ids(json.loads(line), self.max_len)
            except Exception:
                r = None
            if r is not None:
                yield r


def _aria_to_ids(instr, point, max_len=1024, image_token=IMAGE_TOKEN):
    """UI grounding: [image] USER: <instruction> ASSISTANT: (cx, cy) — normalized 0-1000 click point."""
    ids, loss = [image_token], [0]
    q = ENC.encode_ordinary("USER: " + (instr or "").strip() + "\nASSISTANT:")
    ids += q; loss += [0] * len(q)
    a = ENC.encode_ordinary(f" ({int(point[0])}, {int(point[1])})") + [EOT]
    ids += a; loss += [1] * len(a)
    return _pack(ids, loss, max_len)


class StreamAria(IterableDataset):
    """Aria-UI desktop grounding: screenshot + instruction -> normalized click point. Reads a PRE-STAGED compact
    jsonl ({img_file, instr, point}) produced by aria_preprocess(), plus the desktop screenshots zip (pre-staged
    on a volume); images opened by name from the zip. No HF/Xet at train time."""
    def __init__(self, jsonl_path, zip_path, rank=0, world=1, max_len=1024):
        self.jsonl_path, self.zip_path = jsonl_path, zip_path
        self.rank, self.world, self.max_len = rank, world, max_len

    def __iter__(self):
        import zipfile
        from PIL import Image
        shard, n = _shard(self.rank, self.world)
        try:
            zf = zipfile.ZipFile(self.zip_path)
            base2name = {}
            for nm in zf.namelist():
                base2name.setdefault(nm.rsplit("/", 1)[-1], nm)
        except Exception:
            return
        idx = 0
        while True:
            try:
                fh = open(self.jsonl_path, encoding="utf-8")
            except Exception:
                return
            for line in fh:
                if idx % n != shard:
                    idx += 1; continue
                idx += 1
                try:
                    ex = json.loads(line)
                    mem = base2name.get(ex["img_file"].rsplit("/", 1)[-1])
                    if mem is None:
                        continue
                    img = Image.open(io.BytesIO(zf.read(mem))).convert("RGB")
                    if max(img.size) > 2048:
                        img.thumbnail((2048, 2048))
                    r = _aria_to_ids(ex["instr"], ex["point"], self.max_len)
                except Exception:
                    continue
                if r is not None:
                    yield img, r[0], r[1]
            fh.close()


def aria_preprocess(json_path, out_jsonl, max_samples=200000):
    """Stream-convert the big Aria desktop json (array of {img_file, elements:[{instructions[], bbox_norm}]})
    into a compact jsonl of {img_file, instr, point:[cx,cy]} (normalized 0-1000), capped at max_samples."""
    import ijson
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out, open(json_path, "rb") as f:
        for rec in ijson.items(f, "item"):
            img = rec.get("img_file")
            if not img:
                continue
            for el in rec.get("elements", []):
                bb = el.get("bbox_norm")
                if not bb or len(bb) < 4:
                    continue
                cx, cy = int((bb[0] + bb[2]) / 2), int((bb[1] + bb[3]) / 2)
                for instr in (el.get("instructions") or []):
                    out.write(json.dumps({"img_file": img, "instr": instr, "point": [cx, cy]}) + "\n")
                    n += 1
                    if n >= max_samples:
                        return n
    return n


def stream_collate_text(batch):
    ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids); B = len(ids)
    pi = torch.full((B, L), EOT, dtype=torch.long); pt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]; pi[b, :n] = ids[b]; pt[b, :n] = tgts[b]
    return pi, pt


def stream_collate_vlm(batch):
    imgs, ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids); B = len(ids)
    pi = torch.full((B, L), EOT, dtype=torch.long); pt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]; pi[b, :n] = ids[b]; pt[b, :n] = tgts[b]
    return list(imgs), pi, pt
