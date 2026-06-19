"""Grammar-constrained tool-call decoding, ported from cactus-compute/needle (model/constrained.py).

A char-level JSON state machine tracks position in the compact output
`[{"name":"TOOL","arguments":{"key":val,...}}]` and MASKS the logits so that:
  - tool NAMES (after `"name":"`)        -> only valid tool names (name trie)
  - argument KEYS (after `{"` / `,"`)    -> only valid param names for the current tool (param trie)
  - argument VALUES                       -> unconstrained (the model's job)
The model still generates everything itself (structure + values); we only restrict the vocabulary during
the name/key spans. This forces valid names + keys (killing param hallucination / wrong-tool / bad-key
errors) WITHOUT the autoregressive disruption that prefilling causes.
"""
from __future__ import annotations

import json
from enum import Enum, auto

import torch

_TS = None        # token_strings: id -> decoded text
_FIRST = None     # first-char -> list of token ids


def _token_data(tok, gpt2_valid):
    global _TS, _FIRST
    if _TS is None:
        _TS = []
        for i in range(gpt2_valid):
            try:
                _TS.append(tok.decode([i]))
            except Exception:
                _TS.append("")
        _FIRST = {}
        for tid, s in enumerate(_TS):
            if s:
                _FIRST.setdefault(s[0], []).append(tid)
    return _TS, _FIRST


class _Trie:
    def __init__(self):
        self.root = {}
    def insert(self, w):
        n = self.root
        for ch in w:
            n = n.setdefault(ch, {})
        n["$"] = True
    def node(self, prefix):
        n = self.root
        for ch in prefix:
            if ch not in n:
                return None
            n = n[ch]
        return n


def _param_names(params):
    if not isinstance(params, dict):
        return []
    props = params.get("properties")
    if isinstance(props, dict):
        return list(props.keys())
    return [k for k, v in params.items() if isinstance(v, dict)]


class _Constraints:
    def __init__(self, tools_json):
        self.names = _Trie()
        self.params = {}
        try:
            tools = json.loads(tools_json)
        except Exception:
            tools = []
        for t in tools if isinstance(tools, list) else []:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            self.names.insert(t["name"])
            pt = _Trie()
            for k in _param_names(t.get("parameters")):
                pt.insert(k)
            self.params[t["name"]] = pt


class _State(Enum):
    FREE = auto()
    IN_NAME = auto()
    IN_ARG_KEY = auto()


class _Machine:
    """Char-level tracker (mirrors needle's JsonStateMachine)."""
    def __init__(self):
        self.state = _State.FREE
        self.buf = ""
        self.cbuf = ""           # accumulated chars inside a constrained span
        self.func = ""
        self.in_args = False
        self.args_depth = 0
        self.depth = 0
        self.in_string = False
        self.escape = False

    def feed(self, text):
        for ch in text:
            self._feed(ch)

    def _feed(self, ch):
        if self.state in (_State.IN_NAME, _State.IN_ARG_KEY):
            if ch == '"':
                if self.state == _State.IN_NAME:
                    self.func = self.cbuf
                self.cbuf = ""
                self.state = _State.FREE
            else:
                self.cbuf += ch
            self.buf += ch
            return
        self.buf += ch
        if self.in_string:
            if self.escape:
                self.escape = False; return
            if ch == "\\":
                self.escape = True; return
            if ch == '"':
                self.in_string = False
            return
        if ch in "{[":
            self.depth += 1
        elif ch in "}]":
            self.depth = max(0, self.depth - 1)
            if ch == "}" and self.in_args and self.depth < self.args_depth:
                self.in_args = False
            return
        if self.buf.endswith('"name":"') and not self.in_args:
            self.state = _State.IN_NAME; self.cbuf = ""; return
        if self.buf.endswith('"arguments":{'):
            self.in_args = True; self.args_depth = self.depth; return
        if self.in_args and self.depth == self.args_depth and self.buf[-2:] in ('{"', ',"'):
            self.state = _State.IN_ARG_KEY; self.cbuf = ""; return
        if ch == '"' and self._value_quote():
            self.in_string = True

    def _value_quote(self):
        for j in range(len(self.buf) - 2, -1, -1):
            c = self.buf[j]
            if c in " \t\n\r":
                continue
            return c == ":"
        return False


def _token_ok(text, node):
    n = node
    for ch in text:
        if ch == '"':
            return "$" in n
        if ch not in n:
            return False
        n = n[ch]
    return True


def _mask(logits, node, ts, first, dev):
    valid_first = set(node.keys()) - {"$"}
    allow = []
    if "$" in node:
        valid_first.add('"')
    for fc in valid_first:
        for tid in first.get(fc, ()):
            if _token_ok(ts[tid], node):
                allow.append(tid)
    if not allow:
        return logits                                  # off-grammar -> fall back to unconstrained
    m = torch.full_like(logits, float("-inf"))
    m[torch.tensor(allow, device=dev)] = logits[torch.tensor(allow, device=dev)]
    return m


@torch.no_grad()
def constrained_tool_gen(llm, tok, dev, prompt, tools_json, gpt2_valid, eot, max_new=128, force=False, **_):
    ts, first = _token_data(tok, gpt2_valid)
    con = _Constraints(tools_json)
    machine = _Machine()
    if force:                                             # force a call: seed '[' so the model can't abstain
        machine.feed("[")                                 # into prose; the machine then guides {"name":"..."}
    cur = torch.tensor([tok.encode_ordinary(prompt + (" [" if force else ""))[-1900:]], device=dev)
    outs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            lg = llm(cur[:, -2048:])[0][:, -1, :].float()[0]
            lg[gpt2_valid:] = float("-inf")
            node = None
            if machine.state == _State.IN_NAME:
                node = con.names.node(machine.cbuf)
            elif machine.state == _State.IN_ARG_KEY:
                pt = con.params.get(machine.func)
                node = pt.node(machine.cbuf) if pt else None
            if node is not None:
                lg = _mask(lg, node, ts, first, dev)
            t = int(lg.argmax(-1).item())
            if t == eot:
                break
            outs.append(t)
            machine.feed(ts[t])
            cur = torch.cat([cur, torch.tensor([[t]], device=dev)], dim=1)
    return (("[" if force else "") + tok.decode(outs)).strip()   # prepend seeded '[' so parse() sees full call


def _free_greedy(llm, tok, dev, prompt, gpt2_valid, eot, max_new):
    cur = torch.tensor([tok.encode_ordinary(prompt)[-2000:]], device=dev)
    outs = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_new):
            lg = llm(cur[:, -2048:])[0][:, -1, :].float()
            lg[:, gpt2_valid:] = float("-inf")
            t = int(lg.argmax(-1).item())
            if t == eot:
                break
            outs.append(t)
            cur = torch.cat([cur, torch.tensor([[t]], device=dev)], dim=1)
    return tok.decode(outs).strip()
