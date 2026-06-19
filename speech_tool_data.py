"""Spoken-agent data: [SPEECH](spoken query) + TOOLS -> tool call -> TOOL result -> text answer.

Teaches the omni model the full loop: hear a request, emit the tool call, then (after the tool result is
fed back) summarize it in words. Built from Nemotron multi-turn trajectories that contain a
user -> assistant(call) -> tool(result) -> assistant(answer) sequence; the user query is TTS'd to audio
(at prep time) and spliced at the SPEECH sentinel. Loss is on the assistant turns (call + answer).
"""
from __future__ import annotations

import base64
import json

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset

from multimodal import SPEECH_TOKEN, IGNORE_INDEX
from tool_data import flatten_tools, _calls_json

ENC = tiktoken.get_encoding("gpt2")
EOT = 50256


def extract_speech_tool(row):
    """Nemotron row -> (query_text, tools_json, post_segments) for one query->call->result->answer loop,
    or None. post_segments = the text AFTER the spoken query, as (text, is_loss) pieces."""
    tools = flatten_tools(row.get("tools", []))
    if not tools:
        return None
    msgs = row.get("messages", [])
    query = call = result = answer = None
    i = 0
    # find: user, then assistant(tool_calls), then tool, then assistant(text)
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "user" and query is None:
            query = m.get("content") or ""
        elif m.get("role") == "assistant" and m.get("tool_calls") and query and call is None:
            call = _calls_json(m["tool_calls"])
        elif m.get("role") == "tool" and call and result is None:
            c = m.get("content", "")
            result = c if isinstance(c, str) else json.dumps(c, separators=(",", ":"))
        elif m.get("role") == "assistant" and result and not m.get("tool_calls") and answer is None:
            answer = m.get("content") or ""
            break
        i += 1
    if not (query and call and result and answer):
        return None
    tools_json = json.dumps(tools, separators=(",", ":"))
    post = [("\nASSISTANT:", 0), (" " + call, 1),
            ("\nTOOL: " + str(result) + "\nASSISTANT:", 0), (" " + answer.strip(), 1)]
    return query.strip(), tools_json, post


def wav_to_b64(wav):
    return base64.b64encode((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()).decode()


def b64_to_wav(s):
    return np.frombuffer(base64.b64decode(s), dtype="<i2").astype(np.float32) / 32767.0


class SpeechToolSFT(Dataset):
    """Items: {wav_b64, tools, post}. Returns (wav, ids, targets) with one SPEECH sentinel for the spoken
    query; loss masked to the assistant turns (call + answer). build_inputs_embeds expands SPEECH."""
    def __init__(self, path: str, max_len: int = 2048):
        self.data = []
        for p in path.split(","):
            with open(p.strip()) as f:
                self.data += [json.loads(l) for l in f]
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        ids, loss = [], []
        pre = ENC.encode_ordinary("TOOLS: " + ex["tools"] + "\n")
        ids += pre; loss += [0] * len(pre)
        ids += [SPEECH_TOKEN]; loss += [0]              # spoken query
        for text, is_loss in ex["post"]:
            t = ENC.encode_ordinary(text)
            if is_loss:
                t = t + [EOT]
            ids += t; loss += [1 if is_loss else 0] * len(t)
        ids, loss = ids[:self.max_len], loss[:self.max_len]
        L = len(ids)
        inp = torch.tensor(ids[:-1], dtype=torch.long)
        tgt = torch.tensor([ids[k + 1] if loss[k + 1] else IGNORE_INDEX for k in range(L - 1)], dtype=torch.long)
        return b64_to_wav(ex["wav_b64"]), inp, tgt


def speech_tool_collate(batch):
    wavs, ids, tgts = zip(*batch)
    L = max(x.shape[0] for x in ids)
    B = len(ids)
    pad_ids = torch.full((B, L), EOT, dtype=torch.long)
    pad_tgt = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    for b in range(B):
        n = ids[b].shape[0]
        pad_ids[b, :n] = ids[b]
        pad_tgt[b, :n] = tgts[b]
    return [w for w in wavs], pad_ids, pad_tgt
