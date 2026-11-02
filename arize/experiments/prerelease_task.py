"""Pre-release check task for `ax experiments run`: fill a respond-prompt variant with the row's `tool_results` and `input`, call gemma3 on Ollama DIRECTLY.

Env: PROMPT_FILE = path of the respond prompt (see arize/README.md) or VARIANT = name (file ~/prompt-variant-<X>.txt); TIMING_LOG = JSONL path for Ollama timings; MODEL (default gemma3:1b-it-qat, the app's answer model);
TEMPERATURE (default 0.7, the app's AI_TEMPERATURE); OLLAMA_URL (in-cluster service, reached through the k3d node because
the host cannot route to ClusterIPs). Generation only: retrieval is not exercised, the page text is already in `tool_results`.
"""
import json
import os
import subprocess

VARIANT = os.getenv("VARIANT", "deployed")
MODEL = os.getenv("MODEL", "gemma3:1b-it-qat")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
def _ollama_url():
    """The Ollama service's cluster IP changes whenever the cluster is rebuilt, so ask the cluster."""
    ip = subprocess.run(["docker", "exec", os.getenv("K3D_NODE", "k3d-enablement-server-0"), "kubectl", "get", "svc", "ollama",
                         "-n", os.getenv("K8S_NAMESPACE", "ai-travel-advisor"), "-o", "jsonpath={.spec.clusterIP}"], capture_output=True, text=True).stdout.strip()
    return f"http://{ip}:11434"


OLLAMA_URL = os.getenv("OLLAMA_URL") or _ollama_url()
NODE = os.getenv("K3D_NODE", "k3d-enablement-server-0")
PROMPT = open(os.path.expanduser(os.getenv("PROMPT_FILE") or f"~/prompt-variant-{VARIANT}.txt")).read()
TIMING_LOG = os.getenv("TIMING_LOG")  # optional JSONL: one line per call with Ollama's own timings


def task(dataset_row) -> str:
    row = dataset_row if isinstance(dataset_row, dict) else vars(dataset_row)
    props = row.get("additional_properties") or {}
    get = lambda k: row.get(k) if row.get(k) is not None else props.get(k)
    prompt = PROMPT.replace("{tool_results}", str(get("tool_results"))).replace("{request}", str(get("input")))
    body = json.dumps({"model": MODEL, "stream": False, "messages": [{"role": "user", "content": prompt}],
                       "options": {"temperature": TEMPERATURE}})
    cmd = ["docker", "exec", "-i", NODE, "sh", "-c",
           f'wget -qO- -T 300 --header="Content-Type: application/json" --post-data="$(cat)" {OLLAMA_URL}/api/chat']
    out = subprocess.run(cmd, input=body, capture_output=True, text=True, timeout=330)
    if out.returncode != 0:
        raise RuntimeError(f"ollama call failed: {out.stderr[:200]}")
    reply = json.loads(out.stdout)
    if TIMING_LOG:
        keys = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")
        with open(TIMING_LOG, "a") as f:
            f.write(json.dumps({"variant": VARIANT, "case_id": get("case_id"), "input": get("input"), **{k: reply.get(k) for k in keys}}) + "\n")
    return reply["message"]["content"].strip()
