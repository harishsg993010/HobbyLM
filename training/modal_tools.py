"""Modal app for tool-use (function-calling) post-training of the MoE LLM on nvidia/Nemotron-Agentic-v1.
Needle-style single-shot: query + tools -> tool call. Eval uses Needle's metrics (tool_eval.score_tool_calls).

  python -m modal run modal_tools.py --action prep        # build train/val pairs (coverage report)
  python -m modal run modal_tools.py --action train       # 8xH100 SFT from the unified model
  python -m modal run modal_tools.py --action eval         # Needle-protocol tool-call metrics
"""
import modal

_deps = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.0", "tiktoken", "huggingface-hub", "numpy", "pyarrow==17.0.0", "gguf", "datasets")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache/hf"})
)
# Only the Python sources are needed in the image (datasets live on the volume).
# Excluding hobby-chat/ (13 GB Rust artifacts), the GGUFs, and other large local files
# keeps the build context small — a 16 GB context upload was getting reset (WinError 10054)
# on this connection before the run could start.
img = _deps.add_local_dir(
    ".", "/root/moe-lab",
    ignore=["hobby-chat/**", "hobby-rs/**", "hobby-rs-cli/**", "needle/**", "acc_gen/**",
            "runs/**", ".git/**", "__pycache__/**", "**/__pycache__/**",
            "*.gguf", "*.bin", "*.wav", "*.jpg", "*.jpeg", "*.png", "*.log", "**/*.pyc"],
)

app = modal.App("moe-tools", image=img)
runs_vol = modal.Volume.from_name("fineweb10B")                       # checkpoints + tools data live here
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF = modal.Secret.from_name("huggingface")

BACKBONE = "/data/runs/500M_ctx2048/model.pt"
TOOLS_DIR = "/data/tools"


@app.function(volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=60 * 60, secrets=[HF])
def export_gguf(run: str = "500M_computer_use"):
    """Convert /data/runs/{run}/model.pt -> hobbylm-arch GGUF on the volume (F32)."""
    import os, sys, subprocess
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    exp = f"/data/runs/{run}/export"; os.makedirs(exp, exist_ok=True)
    out = f"{exp}/{run}-hobbylm.gguf"
    subprocess.run([sys.executable, "export/to_gguf.py", "--ckpt", f"/data/runs/{run}/model.pt", "--out", out], check=True)
    runs_vol.commit()
    print(f"GGUF -> {out} ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)
    return {"out": out, "bytes": os.path.getsize(out)}


@app.function(image=img, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=2 * 60 * 60, secrets=[HF])
def prep(max_len: int = 2048, val_size: int = 2000, repo: str = "nvidia/Nemotron-Agentic-v1"):
    """Download tool_calling.jsonl, extract single-shot (query+tools -> call) pairs that fit max_len,
    split train/val, write to the volume. Reports coverage (the interactive_agent subset needs >2048 ctx
    so it is intentionally skipped)."""
    import os, sys, json, random
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from huggingface_hub import hf_hub_download
    from hobbylm.tool_data import prep_pairs
    path = hf_hub_download(repo, "data/tool_calling.jsonl", repo_type="dataset", local_dir="/cache/nemotron")
    print(f"downloaded {path} ({os.path.getsize(path)/1e9:.2f} GB)", flush=True)
    with open(path, encoding="utf-8") as f:
        pairs, stats = prep_pairs(f, max_len=max_len)
    cov = 100 * stats["kept"] / max(stats["total"], 1)
    print(f"coverage: kept={stats['kept']} / total={stats['total']} ({cov:.1f}%) | "
          f"no_call={stats['no_call']} too_long={stats['too_long']}", flush=True)
    random.Random(0).shuffle(pairs)
    val, train = pairs[:val_size], pairs[val_size:]
    os.makedirs(TOOLS_DIR, exist_ok=True)
    for name, rows in [("tools_train.jsonl", train), ("tools_val.jsonl", val)]:
        with open(f"{TOOLS_DIR}/{name}", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    runs_vol.commit()
    print(f"wrote train={len(train)} val={len(val)} -> {TOOLS_DIR}", flush=True)
    return {**stats, "train": len(train), "val": len(val)}


@app.function(image=img, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=3 * 60 * 60, secrets=[HF])
def prep_src(source: str, repo: str, files: str, val_size: int = 1000, max_len: int = 2048):
    """Generic prep for the cleaner parquet tool datasets (bitagent / interstellar / ...): read parquet by
    row-group, extract single-shot pairs, filter to max_len, write {source}_train/val.jsonl."""
    import os, sys, json, random
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from hobbylm.tool_data import prep_rows, EXTRACTORS
    ext = EXTRACTORS[source]
    allpairs = []
    agg = {"total": 0, "no_call": 0, "too_long": 0, "kept": 0}
    for f in files.split(","):
        path = hf_hub_download(repo, f, repo_type="dataset", local_dir=f"/cache/{source}")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(rg).to_pylist()
            pairs, st = prep_rows(rows, ext, max_len=max_len)
            allpairs += pairs
            for k in agg:
                agg[k] += st[k]
    cov = 100 * agg["kept"] / max(agg["total"], 1)
    print(f"[{source}] {repo} coverage: kept={agg['kept']}/{agg['total']} ({cov:.1f}%) "
          f"no_call={agg['no_call']} too_long={agg['too_long']}", flush=True)
    random.Random(0).shuffle(allpairs)
    val, train = allpairs[:val_size], allpairs[val_size:]
    os.makedirs(TOOLS_DIR, exist_ok=True)
    for name, rows in [(f"{source}_train.jsonl", train), (f"{source}_val.jsonl", val)]:
        with open(f"{TOOLS_DIR}/{name}", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    runs_vol.commit()
    print(f"[{source}] wrote train={len(train)} val={len(val)} -> {TOOLS_DIR}", flush=True)
    return {**agg, "train": len(train), "val": len(val)}


# Chat SFT: smol-smoltalk (full) + smoltalk2's SFT *_no_think conversational subsets (NOT the 3B reasoning
# Mid subsets, NOT preference/DPO, NOT tool/think). English chat for a "really good chat" 500M.
SMOLTALK2_CHAT = [
    "smoltalk_smollm3_smol_magpie_ultra_no_think", "OpenHermes_2.5_no_think",
    "smoltalk_smollm3_systemchats_30k_no_think", "smoltalk_smollm3_everyday_conversations_no_think",
    "smoltalk_smollm3_explore_instruct_rewriting_no_think", "smoltalk_smollm3_smol_rewrite_no_think",
    "smoltalk_smollm3_smol_summarize_no_think", "tulu_3_sft_personas_instruction_following_no_think",
    "table_gpt_no_think", "Mixture_of_Thoughts_science_no_think",
]


@app.function(image=img, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=6 * 60 * 60,
              memory=32768, cpu=8.0, secrets=[HF])
def prep_chat(max_len: int = 2048, val_size: int = 4000):
    """Build chat SFT trajectories from smol-smoltalk (full) + smoltalk2 SFT no_think conversational
    subsets. Each conversation -> {segments} (loss on assistant turns). Writes chat_{train,val}.jsonl."""
    import os, sys, json, random
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from datasets import load_dataset
    from hobbylm.tool_data import prep_chat as build

    examples, agg = [], {"total": 0, "none": 0, "too_long": 0, "kept": 0}

    def ingest(label, ds):
        msgs = (r.get("messages") for r in ds)
        kept, st = build(msgs, max_len=max_len)
        examples.extend(kept)
        for k in agg:
            agg[k] += st[k]
        print(f"  [{label}] kept {st['kept']}/{st['total']} (none={st['none']} too_long={st['too_long']})", flush=True)

    print("loading smol-smoltalk (train)…", flush=True)
    ingest("smol-smoltalk", load_dataset("HuggingFaceTB/smol-smoltalk", split="train"))
    for sub in SMOLTALK2_CHAT:
        try:
            print(f"loading smoltalk2/SFT/{sub}…", flush=True)
            ingest(sub, load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=sub))
        except Exception as e:
            print(f"  [{sub}] SKIP: {str(e)[:120]}", flush=True)

    random.Random(0).shuffle(examples)
    val, train = examples[:val_size], examples[val_size:]
    os.makedirs(TOOLS_DIR, exist_ok=True)
    for name, rows in [("chat_train.jsonl", train), ("chat_val.jsonl", val)]:
        with open(f"{TOOLS_DIR}/{name}", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    runs_vol.commit()
    print(f"CHAT prep: kept={agg['kept']} total={agg['total']} -> train={len(train)} val={len(val)}", flush=True)
    return {**agg, "train": len(train), "val": len(val)}


@app.function(image=img, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=2 * 60 * 60,
              memory=16384, secrets=[HF])
def prep_dpo(max_len: int = 1024, val_size: int = 2000, repo: str = "HuggingFaceH4/ultrafeedback_binarized"):
    """Build DPO preference pairs from UltraFeedback (what SmolLM2 used post-SFT). Each row ->
    {prompt:'USER: ..\\nASSISTANT:', chosen:' <ans>', rejected:' <ans>'}. Writes dpo_{train,val}.jsonl."""
    import os, sys, json, random
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from datasets import load_dataset
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(repo, split="train_prefs")
    out, total, skip = [], 0, 0
    for r in ds:
        total += 1
        try:
            ch = r["chosen"][-1]["content"]
            rj = r["rejected"][-1]["content"]
            pr = r.get("prompt") or (r["chosen"][0]["content"] if r["chosen"] else "")
        except Exception:
            skip += 1; continue
        if not ch.strip() or not rj.strip() or ch.strip() == rj.strip() or not pr.strip():
            skip += 1; continue
        prompt = f"USER: {pr}\nASSISTANT:"
        # length filter on the longer of the two full sequences
        plen = len(enc.encode_ordinary(prompt))
        if plen + max(len(enc.encode_ordinary(ch)), len(enc.encode_ordinary(rj))) + 1 > max_len:
            skip += 1; continue
        out.append({"prompt": prompt, "chosen": " " + ch, "rejected": " " + rj})
    random.Random(0).shuffle(out)
    val, train = out[:val_size], out[val_size:]
    os.makedirs(TOOLS_DIR, exist_ok=True)
    for name, rows in [("dpo_train.jsonl", train), ("dpo_val.jsonl", val)]:
        with open(f"{TOOLS_DIR}/{name}", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
    runs_vol.commit()
    print(f"DPO prep: kept={len(out)}/{total} (skip={skip}) -> train={len(train)} val={len(val)}", flush=True)
    return {"kept": len(out), "total": total, "train": len(train), "val": len(val)}


@app.function(image=img, gpu="H100:8", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def dpo(max_steps: int = 2000, micro: int = 2, lr: float = 5e-7, beta: float = 0.1,
        save_name: str = "500M_chat_dpo", init_run: str = "500M_chat_v2", train_file: str = "dpo_train.jsonl"):
    """8xH100 DPO from a chat-SFT checkpoint on the preference pairs."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    train_arg = ",".join(f"{TOOLS_DIR}/{p.strip()}" for p in train_file.split(","))
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "training/train_dpo.py",
           "--backbone", BACKBONE, "--init", f"/data/runs/{init_run}/model.pt",
           "--train", train_arg, "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr), "--beta", str(beta)]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=img, volumes={"/data": runs_vol, "/cache/hf": hf_cache}, timeout=3 * 60 * 60, secrets=[HF])
def prep_traj(source: str = "nemotron", val_size: int = 2000, max_len: int = 2048):
    """Extract FULL multi-turn trajectories (loss on every assistant turn) from a source -> {source}_traj_*.jsonl."""
    import os, sys, json, random
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    from huggingface_hub import hf_hub_download
    from hobbylm.tool_data import prep_trajectories
    pairs, agg = [], {"total": 0, "none": 0, "too_long": 0, "kept": 0}
    if source == "nemotron":
        path = hf_hub_download("nvidia/Nemotron-Agentic-v1", "data/tool_calling.jsonl",
                               repo_type="dataset", local_dir="/cache/nemotron")
        with open(path, encoding="utf-8") as f:
            pairs, agg = prep_trajectories(f, "nemotron", max_len=max_len)
    elif source == "bitagent":
        import pyarrow.parquet as pq
        path = hf_hub_download("BitAgent/tool_calling", "data/train-00000-of-00001.parquet",
                               repo_type="dataset", local_dir="/cache/bitagent")
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            p, s = prep_trajectories(pf.read_row_group(rg).to_pylist(), "bitagent", max_len=max_len)
            pairs += p
            for k in agg:
                agg[k] += s[k]
    cov = 100 * agg["kept"] / max(agg["total"], 1)
    print(f"[traj:{source}] kept={agg['kept']}/{agg['total']} ({cov:.1f}%) none={agg['none']} too_long={agg['too_long']}", flush=True)
    random.Random(0).shuffle(pairs)
    val, train_ = pairs[:val_size], pairs[val_size:]
    os.makedirs(TOOLS_DIR, exist_ok=True)
    for name, rows in [(f"{source}_traj_train.jsonl", train_), (f"{source}_traj_val.jsonl", val)]:
        with open(f"{TOOLS_DIR}/{name}", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    runs_vol.commit()
    print(f"[traj:{source}] wrote train={len(train_)} val={len(val)}", flush=True)
    return {**agg, "train": len(train_), "val": len(val)}


@app.function(image=img, gpu="H100:8", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=12 * 60 * 60, secrets=[HF])
def train(max_steps: int = 4000, micro: int = 8, lr: float = 2e-5, save_name: str = "500M_vlm_tools",
          init_run: str = "500M_vlm_joint5", weighted: bool = False, train_file: str = "tools_train.jsonl",
          traj: bool = False, diffusion: bool = False, backbone_run: str = "", max_len: int = 0):
    """8xH100 torchrun SFT: init the LLM from the unified model and tool-tune on the prepared pairs.
    weighted=True -> Needle's name3x/value2x/key1.5x weighted CE (targets argument-value precision).
    diffusion=True -> masked-diffusion (LLaDA) SFT; use backbone_run=<diffusion ckpt> + init_run="none"."""
    import os, subprocess
    os.chdir("/root/moe-lab")
    out = f"/data/runs/{save_name}"
    backbone = f"/data/runs/{backbone_run}/model.pt" if backbone_run else BACKBONE
    train_arg = ",".join(f"{TOOLS_DIR}/{p.strip()}" for p in train_file.split(","))  # combine sources
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "training/train_tools.py",
           "--backbone", backbone, "--train", train_arg, "--out", out,
           "--max_steps", str(max_steps), "--micro", str(micro), "--lr", str(lr)]
    if init_run and init_run.lower() != "none":
        cmd += ["--init", f"/data/runs/{init_run}/model.pt"]
    if max_len:
        cmd += ["--max_len", str(max_len)]
    if diffusion:
        cmd += ["--diffusion", "1", "--traj", "1"]    # chat trajectories, masked-diffusion objective
    elif traj:
        cmd += ["--traj", "1"]
    if weighted:
        cmd += ["--weighted", "1"]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    return {"out": out, "steps": max_steps}


@app.function(image=img, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=3 * 60 * 60, secrets=[HF])
def evaluate(run: str = "500M_vlm_tools", n: int = 400, max_new: int = 96, constrained: bool = True,
             val_file: str = "tools_val.jsonl", backbone: str = ""):
    """Generate tool calls on held-out val; score with Needle's metrics. constrained=True uses the
    schema-constrained decoder (tool name forced to a valid tool, JSON structure guided)."""
    import os, sys, json, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from hobbylm.generate import load_model, GPT2_VALID, EOT
    from hobbylm.tool_eval import score_tool_calls
    from hobbylm.tool_decode import constrained_tool_gen, _free_greedy

    dev = torch.device("cuda")
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt" if backbone else BACKBONE, dev)
    ck = torch.load(f"/data/runs/{run}/model.pt", map_location=dev, weights_only=False)
    llm.load_state_dict(ck["model"]); llm.eval()
    tok = tiktoken.get_encoding("gpt2")
    data = [json.loads(l) for l in open(f"{TOOLS_DIR}/{val_file}")][:n]
    print(f"eval {run} on {len(data)} from {val_file} | constrained={constrained}", flush=True)

    def gen(prompt, tools_json):
        if constrained:
            return constrained_tool_gen(llm, tok, dev, prompt, tools_json, GPT2_VALID, EOT, max_new=max_new)
        return _free_greedy(llm, tok, dev, prompt, GPT2_VALID, EOT, max_new)

    t0 = time.time()
    preds = []
    for i, d in enumerate(data):
        preds.append(gen(d["prompt"], d["tools"]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(data)} ({(time.time()-t0)/(i+1)*1000:.0f}ms/ex)", flush=True)
    m = score_tool_calls([d["answers"] for d in data], preds, [d["tools"] for d in data])

    print("\n=== TOOL-CALL METRICS (Needle protocol) ===", flush=True)
    for k in ("num_samples", "json_parse_rate", "name_f1", "args_acc", "value_acc",
              "param_haluc", "param_miss", "call_f1", "exact_match"):
        v = m[k]
        print(f"  {k:16s} {v:.0f}" if k == "num_samples" else f"  {k:16s} {v*100:6.2f}%", flush=True)
    for d, p in list(zip(data, preds))[:5]:
        q = d["prompt"].split("\n")[-2][:90] if "\n" in d["prompt"] else d["prompt"][:90]
        print(f"\n  Q: {q}\n  R: {d['answers'][:130]}\n  P: {p[:130]}", flush=True)
    try:
        with open(f"/data/runs/{run}/tool_eval.json", "w") as f:
            json.dump(m, f, indent=2)
        runs_vol.commit()
    except Exception as e:
        print(f"(save failed: {e})", flush=True)
    return m


_SRC = {"bitagent": ("BitAgent/tool_calling", "data/train-00000-of-00001.parquet"),
        "interstellar": ("interstellarninja/tool-calls-singleturn", "data/train-00000-of-00001.parquet")}


@app.function(image=img, gpu="H100", volumes={"/data": runs_vol, "/cache/hf": hf_cache},
              timeout=5 * 60 * 60, secrets=[HF])
def bfcl(run: str = "500M_vlm_tools_all", limit: int = 150, max_new: int = 160,
         cats: str = "", debug: int = 0, backbone: str = "", force: int = 0, flat: int = 1):
    """Evaluate on the Berkeley Function Calling Leaderboard (BFCL v3) AST categories + relevance/irrelevance.
    Grammar-constrained decode forces valid BFCL function names. `limit` caps items per category."""
    import os, sys, json, time, torch
    os.chdir("/root/moe-lab"); sys.path.insert(0, "/root/moe-lab")
    import tiktoken
    from huggingface_hub import hf_hub_download
    from hobbylm.generate import load_model, GPT2_VALID, EOT
    from hobbylm.tool_decode import constrained_tool_gen
    from hobbylm.bfcl_eval import score_item

    dev = torch.device("cuda")
    llm, cfg, _, _ = load_model(f"/data/runs/{backbone}/model.pt" if backbone else BACKBONE, dev)
    ck = torch.load(f"/data/runs/{run}/model.pt", map_location=dev, weights_only=False)
    llm.load_state_dict(ck["model"]); llm.eval()
    tok = tiktoken.get_encoding("gpt2")
    REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

    CATS = ["simple", "multiple", "parallel", "parallel_multiple",
            "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
            "live_relevance", "irrelevance", "live_irrelevance"]
    if cats:
        CATS = [c for c in cats.split(",") if c]

    def load_cat(cat):
        q = [json.loads(l) for l in open(hf_hub_download(REPO, f"BFCL_v3_{cat}.json", repo_type="dataset"), encoding="utf-8")]
        ans = {}
        if not cat.endswith("relevance"):
            for l in open(hf_hub_download(REPO, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset"), encoding="utf-8"):
                a = json.loads(l); ans[a["id"]] = a["ground_truth"]
        return q, ans

    def fmt_tools(funcs):
        out = []
        for f in funcs:
            p = f.get("parameters", {}) or {}
            if flat and isinstance(p, dict) and p.get("type") == "object":
                # BFCL gives JSON-Schema params; our model trained on needle's FLAT shape
                # ({param:{type,description,required}}). Convert so eval matches training.
                req = set(p.get("required", []) or [])
                props = p.get("properties", {}) or {}
                p = {k: {"type": (v or {}).get("type", "string"),
                         "description": (v or {}).get("description", "") or "",
                         "required": k in req}
                     for k, v in props.items() if isinstance(v, dict)}
            out.append({"name": f.get("name"), "description": f.get("description", "") or "", "parameters": p})
        return json.dumps(out, separators=(",", ":"))

    def get_query(question):
        msgs = question[0] if question and isinstance(question[0], list) else question
        return " ".join(m.get("content", "") for m in msgs if m.get("role") in ("user", "system"))

    def parse(text):
        try:
            c = json.loads(text)
        except Exception:
            return []
        if isinstance(c, dict):
            c = [c]
        return c if isinstance(c, list) else []

    results, t0 = {}, time.time()
    for cat in CATS:
        try:
            q, ans = load_cat(cat)
        except Exception as e:
            print(f"  {cat}: skip ({e})", flush=True); continue
        if limit:
            q = q[:limit]
        tot, n = 0.0, 0
        for item in q:
            tools_json = fmt_tools(item["function"])
            prompt = f"TOOLS: {tools_json}\nUSER: {get_query(item['question'])}\nASSISTANT:"
            pred = constrained_tool_gen(llm, tok, dev, prompt, tools_json, GPT2_VALID, EOT, max_new=max_new,
                                        force=bool(force))
            sc = score_item(cat, parse(pred), ans.get(item["id"]))
            tot += sc; n += 1
            if debug and n <= debug:
                print(f"  [{item['id']}] sc={sc}\n    Q: {get_query(item['question'])[:90]}\n"
                      f"    GT: {json.dumps(ans.get(item['id']))[:160]}\n    P:  {pred[:160]}", flush=True)
        results[cat] = (tot / max(n, 1), n)
        print(f"  {cat:26s} {tot/max(n,1)*100:6.2f}%  (n={n})  [{(time.time()-t0):.0f}s]", flush=True)

    print("\n=== BFCL v3 (our model, AST/grammar-constrained) ===", flush=True)
    for cat, (acc, n) in results.items():
        print(f"  {cat:26s} {acc*100:6.2f}%  (n={n})", flush=True)
    try:
        with open(f"/data/runs/{run}/bfcl.json", "w") as f:
            json.dump({c: {"acc": a, "n": n} for c, (a, n) in results.items()}, f, indent=2)
        runs_vol.commit()
    except Exception:
        pass
    return results


@app.local_entrypoint()
def main(action: str = "prep", max_steps: int = 4000, micro: int = 8, lr: float = 2e-5,
         run: str = "500M_vlm_tools", n: int = 400, constrained: int = 1, weighted: int = 0,
         source: str = "", val_file: str = "tools_val.jsonl", save: str = "", init: str = "500M_vlm_joint5",
         cats: str = "", debug: int = 0, backbone: str = "", force: int = 0, train_file: str = "", flat: int = 1,
         traj: int = 0):
    if action == "export":
        export_gguf.remote(run=run)
    elif action == "prep_chat":
        prep_chat.remote()
    elif action == "prep_dpo":
        prep_dpo.remote()
    elif action == "dpo":
        dpo.remote(max_steps=(max_steps if max_steps != 4000 else 2000), micro=micro, lr=(lr if lr != 2e-5 else 5e-7),
                   save_name=(save or "500M_chat_dpo"), init_run=(init if init != "500M_vlm_joint5" else "500M_chat_v2"))
    elif action == "prep":
        prep.remote()
    elif action == "prep_src":
        repo, f = _SRC[source]
        prep_src.remote(source=source, repo=repo, files=f, val_size=(400 if source == "interstellar" else 2000))
    elif action == "train":
        train.remote(max_steps=max_steps, micro=micro, lr=lr, weighted=bool(weighted), traj=bool(traj),
                     save_name=(save or "500M_vlm_tools"), init_run=init,
                     train_file=(train_file or (f"{source}_train.jsonl" if source else "tools_train.jsonl")))
    elif action == "train_weighted":
        train.remote(max_steps=max_steps, micro=micro, lr=lr, save_name="500M_vlm_tools_w", weighted=True)
    elif action == "prep_traj":
        prep_traj.remote(source=(source or "nemotron"))
    elif action == "train_diff":
        # masked-diffusion (LLaDA) chat SFT of the diffusion base model on smoltalk trajectories
        train.remote(max_steps=(max_steps if max_steps != 4000 else 8000), micro=(micro if micro != 8 else 16),
                     lr=(lr if lr != 2e-5 else 2e-5), diffusion=True, backbone_run=(backbone or "500M_diff_20b"),
                     init_run="none", save_name=(save or "500M_diff_chat"),
                     train_file=(train_file or "chat_train.jsonl"), max_len=1024)
    elif action == "train_traj":
        train.remote(max_steps=(max_steps if max_steps != 4000 else 6000), micro=micro, lr=lr, traj=True,
                     save_name=(save or "500M_vlm_tools_traj"),
                     init_run=(init if init != "500M_vlm_joint5" else "500M_vlm_tools_all"),
                     train_file=(f"{source}_traj_train.jsonl" if source
                                 else "nemotron_traj_train.jsonl,bitagent_traj_train.jsonl"))
    elif action == "train_bal":
        # balanced: single-shot must-call (anchors "call when relevant") + multi-turn trajectories (chain+abstain)
        train.remote(max_steps=(max_steps if max_steps != 4000 else 6000), micro=micro, lr=lr, traj=True,
                     save_name=(save or "500M_vlm_tools_bal"), init_run="500M_vlm_tools_all",
                     train_file="tools_train.jsonl,bitagent_train.jsonl,nemotron_traj_train.jsonl")
    elif action == "train_combined":
        train.remote(max_steps=(max_steps if max_steps != 4000 else 6000), micro=micro, lr=lr, weighted=True,
                     save_name=(save or "500M_vlm_tools_all"), init_run=init,
                     train_file="tools_train.jsonl,bitagent_train.jsonl,interstellar_train.jsonl")
    elif action == "eval":
        evaluate.remote(run=run, n=n, constrained=bool(constrained), val_file=val_file, backbone=backbone)
    elif action == "bfcl":
        bfcl.remote(run=run, limit=(n if n != 400 else 150), cats=cats, debug=debug, backbone=backbone, force=force, flat=flat)
    else:
        raise SystemExit(f"unknown action {action!r} (use prep|prep_src|train|train_weighted|eval|bfcl)")
