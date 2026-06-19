"""Tool-call metrics ported from cactus-compute/needle (needle/training/eval.py:benchmark_tool_calls),
so our function-calling numbers are directly comparable to Needle's.

score_tool_calls(refs, preds, tools) takes lists of JSON strings:
  refs  = ground-truth answers  '[{"name","arguments"}]'
  preds = model output text
  tools = available tools        '[{"name","description","parameters"}]'
Returns the same metric dict Needle prints: exact_match, json_parse_rate, name_f1, args_acc, call_f1,
param_haluc, param_miss, value_acc.
"""
from __future__ import annotations

import json
import re


def to_snake_case(name):
    if not isinstance(name, str):
        return name
    s = re.sub(r"[\s\-]+", "_", name.strip())
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return s.lower()


def _norm_value(v):
    if isinstance(v, str):
        try:
            v = str(float(v))
        except ValueError:
            pass
        s = v.strip().lower()
        if s.startswith("at "):
            s = s[3:].strip()
        if s.startswith("today at "):
            s = s[len("today at "):].strip()
        return s
    if isinstance(v, float):
        return str(v)
    return v


def _norm_args(args):
    if not isinstance(args, dict):
        return args
    return {k: _norm_value(v) for k, v in args.items()}


def _call_key(c):
    if not isinstance(c, dict):
        return None
    return json.dumps({"name": c.get("name"), "arguments": _norm_args(c.get("arguments", {}))}, sort_keys=True)


def score_tool_calls(refs, preds, tools):
    total = exact = jperr = 0
    tp_n = fp_n = fn_n = tp_c = fp_c = fn_c = 0
    args_correct = args_total = 0
    halluc = total_pred_p = missing = total_ref_p = correct_val = matched = 0

    for ref_text, pred_text, tools_text in zip(refs, preds, tools):
        total += 1
        pred_text = (pred_text or "").strip()
        try:
            ref_calls = json.loads(ref_text)
        except (json.JSONDecodeError, TypeError):
            ref_calls = []
        for rc in ref_calls:
            if isinstance(rc, dict) and "name" in rc:
                rc["name"] = to_snake_case(rc["name"])
        try:
            pred_calls = json.loads(pred_text)
            if isinstance(pred_calls, dict):
                pred_calls = [pred_calls]
            if not isinstance(pred_calls, list):
                pred_calls = []
            for pc in pred_calls:
                if isinstance(pc, dict) and "name" in pc:
                    pc["name"] = to_snake_case(pc["name"])
        except (json.JSONDecodeError, TypeError):
            jperr += 1
            pred_calls = []

        ref_empty = ref_text.strip() in ("", "[]")
        pred_empty = pred_text.strip() in ("", "[]")
        if ref_empty and pred_empty:
            exact += 1
        elif not ref_empty and not pred_empty:
            rk = sorted(k for k in (_call_key(c) for c in ref_calls) if k)
            pk = sorted(k for k in (_call_key(c) for c in pred_calls) if k)
            if rk == pk and len(rk) == len(ref_calls) and len(pk) == len(pred_calls):
                exact += 1

        ref_names = {c["name"] for c in ref_calls if isinstance(c, dict) and "name" in c}
        pred_names = {c["name"] for c in pred_calls if isinstance(c, dict) and "name" in c}
        tp_n += len(pred_names & ref_names); fp_n += len(pred_names - ref_names); fn_n += len(ref_names - pred_names)

        rkeys = {k for k in (_call_key(c) for c in ref_calls) if k}
        pkeys = {k for k in (_call_key(c) for c in pred_calls) if k}
        tp_c += len(pkeys & rkeys); fp_c += len(pkeys - rkeys); fn_c += len(rkeys - pkeys)

        ref_by_name = {}
        for c in ref_calls:
            if isinstance(c, dict) and "name" in c:
                ref_by_name.setdefault(c["name"], []).append(c.get("arguments", {}))
        for c in pred_calls:
            if isinstance(c, dict) and c.get("name") in ref_by_name:
                args_total += 1
                pa = json.dumps(_norm_args(c.get("arguments", {})), sort_keys=True)
                if any(pa == json.dumps(_norm_args(ra), sort_keys=True) for ra in ref_by_name[c["name"]]):
                    args_correct += 1

        def _pkeys(params):
            if not isinstance(params, dict):
                return set()
            props = params.get("properties")
            if isinstance(props, dict):
                return set(props.keys())
            return {k for k in params.keys() if k not in ("type", "required", "properties", "description")}

        try:
            tdefs = json.loads(tools_text)
            tpm = {to_snake_case(t["name"]): _pkeys(t.get("parameters"))
                   for t in tdefs if isinstance(t, dict) and "name" in t}
        except (json.JSONDecodeError, TypeError):
            tpm = {}
        for c in pred_calls:
            if not isinstance(c, dict) or c.get("name") not in tpm:
                continue
            _a = c.get("arguments")
            pk_ = set(_a.keys()) if isinstance(_a, dict) else set()
            total_pred_p += len(pk_); halluc += len(pk_ - tpm[c["name"]])
            if c["name"] in ref_by_name:
                rargs = ref_by_name[c["name"]][0]
                rk_ = set((rargs if isinstance(rargs, dict) else {}).keys())
                total_ref_p += len(rk_); missing += len(rk_ - pk_)
                for k in (pk_ & rk_):
                    matched += 1
                    if json.dumps(_norm_value(c["arguments"][k]), sort_keys=True) == json.dumps(_norm_value(rargs[k]), sort_keys=True):
                        correct_val += 1

    def f1(tp, fp, fn):
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-9)

    return {
        "num_samples": total,
        "exact_match": exact / max(total, 1),
        "json_parse_rate": 1.0 - jperr / max(total, 1),
        "name_f1": f1(tp_n, fp_n, fn_n),
        "args_acc": args_correct / max(args_total, 1),
        "call_f1": f1(tp_c, fp_c, fn_c),
        "param_haluc": halluc / max(total_pred_p, 1),
        "param_miss": missing / max(total_ref_p, 1),
        "value_acc": correct_val / max(matched, 1),
    }
