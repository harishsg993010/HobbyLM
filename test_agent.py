"""Test agent loop over the hobby-chat OpenAI API: can the 500M model use the computer via a tool?

It exposes ONE tool — run_command(command) — and runs an agentic loop: the model decides to call the
tool, we execute the shell command, feed the result back, and the model answers. This checks whether the
model can (a) pick the tool, (b) emit a sensible command, and (c) use the output to answer.
"""
import json, subprocess
from urllib.request import Request, urlopen

API = "http://127.0.0.1:11250/v1/chat/completions"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command on the user's computer and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]

DENY = ["rm ", "rmdir", "del ", "remove-item", "format", "mkfs", "shutdown", "reboot",
        "deltree", "rd /s", "> /dev", ":(){", "diskpart"]


def run_command(cmd: str) -> str:
    if not cmd.strip():
        return "(empty command)"
    if any(d in cmd.lower() for d in DENY):
        return "[blocked: potentially destructive command]"
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=15)
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        return out[:1000] if out else "(no output)"
    except Exception as e:
        return f"[error: {e}]"


def call_api(messages):
    body = json.dumps({
        "model": "moe-omni-500m", "messages": messages, "tools": TOOLS,
        "max_tokens": 160, "temperature": 0,
    }).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    return json.load(urlopen(req, timeout=180))["choices"][0]["message"]


def agent(task, max_steps=4):
    print(f"\n=== TASK: {task} ===")
    messages = [{"role": "user", "content": task}]
    last_output = None
    for step in range(1, max_steps + 1):
        msg = call_api(messages)
        tcs = msg.get("tool_calls")
        if tcs:
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tcs})
            for tc in tcs:
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                cmd = args.get("command", "")
                print(f"  step {step}: TOOL run_command  command={cmd!r}")
                last_output = run_command(cmd)
                print(f"           output: {last_output[:160]!r}")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": last_output})
        else:
            content = (msg.get("content") or "").strip()
            # the 500M often emits a call-shaped blob instead of summarizing — fall back to the tool output
            if content.startswith(("[", "{")) and last_output is not None:
                print(f"  step {step}: (model didn't summarize; using the tool output)")
                print(f"  RESULT: {last_output[:200]}")
            else:
                print(f"  step {step}: ANSWER: {content}")
            return
    if last_output is not None:
        print(f"  RESULT (raw tool output): {last_output[:200]}")


if __name__ == "__main__":
    for t in [
        "List the files in the current directory.",
        "What is the current date and time on this computer?",
        "Show the current working directory path.",
    ]:
        agent(t)
