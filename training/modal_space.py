"""Create + push the HobbyLM Playground Gradio Space to Hugging Face (uses the `huggingface` Modal
secret). Bundles the local `space/` dir (app.py + vendored hobbylm/ + hobby_image/) into the image.

  python -m modal run training/modal_space.py                      # create + upload the Space
  python -m modal run training/modal_space.py --action zerogpu     # request ZeroGPU hardware
"""
import modal

app = modal.App("hobbylm-space-push")

HF_USER = "rootxhacker"
SPACE_REPO = f"{HF_USER}/HobbyLM-Playground"

img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("huggingface_hub>=0.25.0", "requests")
       .add_local_dir("space", "/root/space"))

HF_SECRET = modal.Secret.from_name("huggingface")


@app.function(image=img, secrets=[HF_SECRET], timeout=15 * 60)
def push(set_zerogpu: bool = False):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path="/root/space", repo_id=SPACE_REPO, repo_type="space",
                      ignore_patterns=["**/__pycache__/**", "*.pyc"])
    if set_zerogpu:
        try:
            api.request_space_hardware(repo_id=SPACE_REPO, hardware="zero-a10g")
            print("requested ZeroGPU hardware", flush=True)
        except Exception as e:
            print(f"(could not set hardware automatically: {e}); set ZeroGPU in Space settings", flush=True)
    print(f"pushed Space -> https://huggingface.co/spaces/{SPACE_REPO}", flush=True)
    return SPACE_REPO


@app.function(image=img, secrets=[HF_SECRET], timeout=5 * 60)
def logs(kind: str = "run"):
    import os, requests
    from huggingface_hub import get_token
    tok = get_token() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    url = f"https://huggingface.co/api/spaces/{SPACE_REPO}/logs/{kind}"
    out = []
    with requests.get(url, headers={"Authorization": f"Bearer {tok}"}, stream=True, timeout=60) as r:
        print("HTTP", r.status_code, flush=True)
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "replace")
            if s.startswith("data: "):
                s = s[6:]
            out.append(s)
            if len(out) > 250:
                break
    print("\n".join(out[-220:]), flush=True)


@app.local_entrypoint()
def main(action: str = "push"):
    if action == "logs":
        logs.remote("run")
    elif action == "build-logs":
        logs.remote("build")
    else:
        print(push.remote(set_zerogpu=(action == "zerogpu")))
