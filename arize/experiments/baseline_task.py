"""Stage 3 task for `ax experiments run`: send each golden-dataset row to the live travel advisor app.

For every dataset example the task calls GET /api/v1/completion with the row's `input` and returns the
answer text (the `message` of the app's JSON response). Arize stores that return value as the run's `output`,
which the evaluators then score against `expected_output` and `key_facts`.

Settings (environment variables, all optional):
  APP_URL         base URL of the app          (default http://localhost: k3d load balancer -> ingress-nginx -> nginx -> app)
  APP_HOST        Host header for the ingress  (default empty; the devcontainer's ingress needs the ingress host, see arize/README.md)
  FRAMEWORK       agentic | rag | llm          (default: agentic)
  RUN_ID          label added to the session id (default: timestamp)
  TASK_TIMEOUT_S  per-request timeout, seconds (default: 300; an agentic answer takes about 40 s)

Each request is sent with session_id "golden-exp-<framework>-<run id>", so the experiment's traces are easy to
find in your tracing project and to keep apart from your own testing.

Run (first a dry run: it executes only the first 10 examples locally and uploads nothing):
  FRAMEWORK=agentic ax experiments run --name baseline-agentic-no-sf-page \
      --dataset san-francisco-experiment-dataset --space <space id> \
      --task arize/experiments/baseline_task.py --concurrency 1 --dry-run
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

APP_URL = os.getenv("APP_URL", "http://localhost").rstrip("/")
APP_HOST = os.getenv("APP_HOST", "")
FRAMEWORK = os.getenv("FRAMEWORK", "agentic")
RUN_ID = os.getenv("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
TIMEOUT_S = int(os.getenv("TASK_TIMEOUT_S", "300"))
ATTEMPTS = 2  # connection problems and timeouts are retried once, then raised (not scored as an answer)


def _extract_input(row):
    """The dataset example's `input` column, whatever shape the runner hands us."""
    value = None
    if isinstance(row, dict):
        value = row.get("input")
        if value is None and isinstance(row.get("additional_properties"), dict):
            value = row["additional_properties"].get("input")
    if value is None:
        value = getattr(row, "input", None)
    if value is None:
        raise ValueError(f"no 'input' found in dataset row: {row!r}")
    return str(value)


def _answer_from(body: bytes) -> str:
    """The text of the app's JSON response: `message` for answers, `error`/`detail` for application errors."""
    text = body.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict):
        for key in ("message", "error", "detail"):
            if isinstance(data.get(key), str):
                return data[key]
            if isinstance(data.get(key), dict) and isinstance(data[key].get("message"), str):
                return data[key]["message"]
    return text


def task(dataset_row) -> str:
    prompt = _extract_input(dataset_row)
    query = urllib.parse.urlencode(
        {
            "framework": FRAMEWORK,
            "prompt": prompt,
            "session_id": f"golden-exp-{FRAMEWORK}-{RUN_ID}",
        }
    )
    url = f"{APP_URL}/api/v1/completion?{query}"

    last_error = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Host": APP_HOST} if APP_HOST else {}), timeout=TIMEOUT_S) as response:
                return _answer_from(response.read())
        except urllib.error.HTTPError as err:
            # The app answered with an error status (for example 400/500): record it as the run's output
            # so the evaluators fail it visibly, instead of hiding it.
            return f"[HTTP {err.code}] {_answer_from(err.read())}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last_error = err
            time.sleep(2 * attempt)
    raise RuntimeError(f"app unreachable after {ATTEMPTS} attempts: {last_error}")
