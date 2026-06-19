"""Berkeley Function Calling Leaderboard (BFCL v3) AST scorer — a faithful-enough port of gorilla's
AST accuracy for the non-exec categories (simple/multiple/parallel/parallel_multiple + live variants)
plus relevance/irrelevance.

Ground truth per item: list of {func_name: {param: [acceptable_value, ...]}}. A predicted call matches
when the function name is right, every predicted param is a real param, and each ground-truth param's
value is in its acceptable list (`""` in the list => the param is optional and may be omitted). Parallel
categories require an order-independent 1:1 match of all calls. relevance => model must call; irrelevance
=> model must NOT call.
"""
from __future__ import annotations


def _norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        low = s.lower()
        if low in ("true", "false"):                 # "true"/"false" strings == booleans (BFCL bug fix)
            return low == "true"
        try:
            return float(s)
        except ValueError:
            return low
    if isinstance(v, list):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _norm(x)) for k, x in v.items()))
    return v


def _param_ok(p, accept, pred_args):
    optional = any(a == "" for a in accept)
    if p not in pred_args:
        return optional
    pv = _norm(pred_args[p])
    return any(pv == _norm(a) for a in accept if a != "" or pv in ("", _norm("")))


def match_call(pred_call, gt_entry):
    if not isinstance(pred_call, dict):
        return False
    fname = next(iter(gt_entry))
    if pred_call.get("name") != fname:
        return False
    gt_params = gt_entry[fname]
    pred_args = pred_call.get("arguments") or {}
    if not isinstance(pred_args, dict):
        return False
    for p in pred_args:                              # no params outside the ground-truth set
        if p not in gt_params:
            return False
    return all(_param_ok(p, accept, pred_args) for p, accept in gt_params.items())


def match_multi(pred_calls, gt_list):
    """Order-independent 1:1 match of all calls (used for parallel / parallel_multiple)."""
    if len(pred_calls) != len(gt_list):
        return False
    used = [False] * len(pred_calls)
    for gt in gt_list:
        hit = False
        for i, pc in enumerate(pred_calls):
            if not used[i] and match_call(pc, gt):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True


def score_item(category, pred_calls, gt):
    """Return 1.0/0.0 for one BFCL item. `gt` is the item's ground_truth list (or None for irrelevance)."""
    pred_calls = [c for c in (pred_calls or []) if isinstance(c, dict)]
    if category.endswith("irrelevance"):
        return 1.0 if len(pred_calls) == 0 else 0.0
    if category.endswith("relevance"):              # live_relevance: a tool SHOULD be called
        return 1.0 if len(pred_calls) > 0 else 0.0
    if not gt:
        return 0.0
    return 1.0 if match_multi(pred_calls, gt) else 0.0
