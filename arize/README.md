# Arize AX constructs for the AI Travel Advisor workshop

Everything the workshop builds in Arize AX, as files and `ax` CLI commands you can run again: evaluators, live evaluation tasks, annotation configs and labeling queues, a dataset, and experiments. It was built and checked by hand in the pilot (a "San Francisco" destination that is missing from the knowledge base, found with evaluators and fixed by adding a page). Nothing here needs the Arize SDK: the scripts call only the `ax` CLI.

## What you need first

The travel advisor must be running and sending traces to an Arize AX project, and the `ax` CLI must be logged in (`ax profiles`). Nothing in this directory is tied to an account, space or project, so you give these values as parameters. Each one is a script option or an environment variable, and the CLI can look it up:

| Variable | Option / env var | Used by | Look it up with the CLI |
|---|---|---|---|
| Space ID | `--space` / `ARIZE_SPACE_ID` | all three scripts | `ax spaces list` (use the **ID**, not the name: two spaces can look alike) |
| Tracing project | `--project` / `ARIZE_PROJECT` | `create_tasks.py` | `ax projects list --space $SPACE` (the name the app sets as `openinference.project.name`) |
| Judge AI integration ID | `--ai-integration-id` / `ARIZE_AI_INTEGRATION_ID` | `create_evaluators.py` | `ax ai-integrations list --space $SPACE` (the `id` field) |
| Judge model | `--model-name` / `ARIZE_JUDGE_MODEL` | `create_evaluators.py` | the integration's `model_names` (Azure: the deployment name): `ax ai-integrations get <integration id> -o json` |
| Judge provider settings | `--provider-params` / `ARIZE_JUDGE_PROVIDER_PARAMS` | `create_evaluators.py` | Azure OpenAI only: the integration's `base_url` is the endpoint; the API version is your choice (the pilot used `2025-04-01-preview`) |
| Annotator email | `--annotator-email` / `ARIZE_ANNOTATOR_EMAIL` | `create_annotations.py` | `ax users list` (your Arize login; a queue needs an annotator) |

The judge's settings are yours, so they are parameters, never files in this repo. The pilot used an Azure OpenAI deployment that accepts only temperature 1, so the same trace can get a different label on a rerun (judge noise).

To fill them all in from the CLI:

```bash
ax spaces list                                   # -> SPACE
ax projects list --space $SPACE                  # -> PROJECT
ax ai-integrations list --space $SPACE           # -> id = AI_INTEGRATION
ax ai-integrations get $AI_INTEGRATION -o json   # -> model_names[0] = JUDGE_MODEL, base_url = Azure endpoint
ax users list                                    # -> ANNOTATOR_EMAIL
```

## What is in this directory

| Path | What |
|---|---|
| `config/evaluators.json`, `config/templates/*.txt` | the six LLM-judge evaluators: pre-built Arize template text, labels, direction, granularity |
| `config/tasks.json` | the six live evaluation tasks: filter, sampling, variable mappings |
| `config/annotations.json` | the three human annotation configs and the two labeling queues (with their instructions) |
| `evaluators/key-facts-coverage.py`, `key-facts-imports.py` | the one code evaluator (offline, experiments only) |
| `datasets/golden-dataset-rows.csv` | the 10 dataset rows (San Francisco, declines, regression) |
| `datasets/build_prerelease_rows.py` | builds the rows for the pre-release check |
| `experiments/baseline_task.py` | experiment task: sends each row to the live app |
| `experiments/prerelease_task.py` | experiment task: calls the answer model directly with a prompt and the row's tool results |
| `create_evaluators.py`, `create_tasks.py`, `create_annotations.py` | create the objects above with the CLI and read them back |

## The evaluators and what each scores

| Evaluator | Granularity | Scored span (task filter) | Labels |
|---|---|---|---|
| `task_completion` | span | root span `name = 'agentic_advice_workflow.workflow'`: input and output | `completed` / `failed` |
| `tool_calling` | span | first tool-choice call (`langgraph_node = 'call_model'`, kind `LLM`, `langgraph_step = '1'`) | `correct` / `incorrect` |
| `step_efficiency` | trace | the trace, matched by its root span; reads the whole trajectory | `efficient` / `redundant` |
| `rag_relevancy` | span | `RETRIEVER` spans | `relevant` / `unrelated` |
| `hallucination` | span | the answer-model span (`langgraph_node = 'respond'`): request, tool results, answer | `factual` / `hallucinated` |
| `toxicity` | span | same answer-model span, 20% sampling | `non-toxic` / `toxic` |
| `key-facts-coverage` (code) | span | offline, on experiment rows: share of the row's `key_facts` found in the answer | score 0 to 1; no key facts = not applicable |

Result columns on the spans: `eval.<name>.label` / `.score` / `.explanation`, and `trace_eval.step_efficiency.*` on the root span.

**Why every task is one filter plus `column_mappings`:** the API has a second task shape (several named filters combined with `A AND B`) that only the SDK and the UI can create. This setup avoids it so the CLI is enough. The price: `tool_calling` judges the first tool-choice call only, and `step_efficiency` joins each variable over every span of the trace (100K-character cap). Trace-granularity evaluators otherwise read every span of the trace, which made `task_completion` fail on day one of the pilot, so it is scored on the root span.

**What breaks silently if the app changes:** the filters use the graph node names `call_model` and `respond`, the step `langgraph_step = '1'`, the agent name `travel_agent` and the root span name `agentic_advice_workflow.workflow`. If any of them changes, the matching task selects nothing.

## Commands

```bash
SPACE=<space id>                 # see "What you need first" for how to look each value up
PROJECT=<tracing project name>
AI_INTEGRATION=<integration id>
JUDGE_MODEL=<judge deployment or model>
ANNOTATOR_EMAIL=<your Arize email>
cd arize

# 1. Evaluators (once per space; add --suffix " (test)" for a test copy; --dry-run shows the commands)
python3 create_evaluators.py --space $SPACE \
  --ai-integration-id $AI_INTEGRATION --model-name $JUDGE_MODEL \
  --provider-params '{"azure_params": {"azure_deployment_name": "<deployment>", "azure_openai_endpoint": "https://<resource>.openai.azure.com", "azure_openai_version": "2025-04-01-preview"}}'

# 2. The six live tasks on the project; each is read back and compared (OK / MISMATCH)
python3 create_tasks.py --space $SPACE --project $PROJECT
```

A continuous task scores only traces that arrive **after** it exists. Create the tasks first, then send traffic. To score traces that already exist (for example to see a burst right away), backfill a window:

```bash
ax tasks list --space $SPACE --limit 100
ax tasks trigger-run <task id> --data-start-time 2026-01-01T10:00:00Z --data-end-time 2026-01-01T10:10:00Z
ax tasks list-runs <task id>      # COMPLETED when done
```

### Read the labels

```bash
# the span export stops at 500 spans: filter per span type
ax spans export $PROJECT --space $SPACE --filter "name = 'agentic_advice_workflow.workflow'" \
   --start-time <ISO UTC> --end-time <ISO UTC> --limit 500 --stdout
```

Each span has an `evaluations` list (`name`, `label`, `score`, `explanation`). `hallucination` and `toxicity` are on the answer-model span, `tool_calling` on the first `call_model` span, `rag_relevancy` on the retriever span, `task_completion` and `step_efficiency` on the root span. A destination the knowledge base does not know is declined without calling the answer model, so it has no `hallucination` label.

### Human review (calibrate the judges, sign off a fix)

```bash
python3 create_annotations.py --space $SPACE --annotator-email $ANNOTATOR_EMAIL
# queues start empty: add answer-model spans in the UI (Traces table > select > Send to Labeling Queue),
# or pass --record-sources <file> with [{"record_type":"SPAN","project_id":"<project id>","start_time":"...Z","end_time":"...Z","span_ids":["<span id>", ...]}]
ax annotation-queues list --space $SPACE --limit 100
ax annotation-queues list-records <queue id> --space $SPACE --limit 100 -o json   # `annotations` = human, `evaluations` = judges
```

Label first and compare after (the judge's label can bias you). In the pilot the judges agreed with the human on `task_completion` 8 of 10 and 8 of 8, and on `hallucination` 7 of 10 and 6 of 8. They were stricter than the human every time, and each time the unsupported detail they cited was not in the facts.

### Dataset and experiments (offline, with `key-facts-coverage`)

```bash
# the dataset (the first version must hold every row, because the "before" experiment scores them)
ax datasets create --name san-francisco-experiment-dataset --space $SPACE --file datasets/golden-dataset-rows.csv

# "before": the live app, agentic only, one request at a time. Dry run first (10 rows, nothing uploaded).
# The devcontainer's ingress needs its host: kubectl -n <app namespace> get ingress -o jsonpath='{.items[0].spec.rules[0].host}'
APP_URL=http://localhost APP_HOST=<ingress host> FRAMEWORK=agentic RUN_ID=before1 \
  ax experiments run --name baseline-agentic-no-sf-page --dataset san-francisco-experiment-dataset --space $SPACE \
  --task experiments/baseline_task.py --concurrency 1 --dry-run
# ... then the same without --dry-run. Use a NEW name every time: a stopped run leaves a hidden experiment that blocks its name.

# score it with the code evaluator (leave the task filter empty; one task can cover several experiments)
ax evaluators list --space $SPACE --limit 100        # note the id of key-facts-coverage
ax tasks create-evaluation --name "key-facts-coverage (baseline)" --task-type CODE_EVALUATION \
  --dataset san-francisco-experiment-dataset --space $SPACE \
  --evaluators '[{"evaluator_id":"<id>","column_mappings":{"output":"output","key_facts":"key_facts"}}]' \
  --experiment-ids <experiment id>
ax tasks trigger-run <task id> --experiment-ids <experiment id>
ax experiments list-runs <experiment id> --dataset san-francisco-experiment-dataset --space $SPACE --limit 100 -o json
#   per-run scores: additional_properties["eval.key-facts-coverage.score"] and ".label" (all_present / none_present)
```

Pilot results for this dataset: before the San Francisco page, its four San Francisco rows score 0.00 (the agent declines), the decline rows are not applicable and the regression rows (Boston, Detroit, Linz) score 1.00. After the page: 0.75 (the typo row `San Fransisco` still declines) and 1.00.

**Pre-release check** (test the draft page offline, before it is in the app): the answer model is called directly with the page text and the deployed answer prompt, so the app gets no traffic.

```bash
python3 - <<'E'
s = open('../app/prompts/agentic_instructions.txt').read()
open('/tmp/respond-prompt.txt', 'w').write(s.split('### RESPOND\n')[1])
E
python3 datasets/build_prerelease_rows.py --sf-page <draft san-francisco.html> > /tmp/prerelease-rows.csv
ax datasets create --name prerelease-check-dataset --space $SPACE --file /tmp/prerelease-rows.csv
PROMPT_FILE=/tmp/respond-prompt.txt VARIANT=deployed \
  ax experiments run --name prerelease-check-deployed --dataset prerelease-check-dataset --space $SPACE \
  --task experiments/prerelease_task.py --concurrency 1
# score it with key-facts-coverage as above
```

`experiments/prerelease_task.py` reaches Ollama through the k3d node (`docker exec`; namespace from `K8S_NAMESPACE`) and looks up the service address itself, because the cluster IP changes whenever the cluster is rebuilt. Run it on the devcontainer host.

## Caveats met while building this

- Names of evaluators and experiments are unique within a space. `ax ... list` shows 15 by default: pass `--limit 100`.
- `ax evaluators create-evaluator template --template` takes the text inline; it does not read `@file` (`--code` and `--imports` do).
- A categorical annotation config with an optimization direction requires scores, which the CLI cannot set: create it with `--optimization-direction NONE`.
- A template evaluation task over several experiments stalls after the first one: make one template task per experiment. A code task can cover several.
- A task run stuck in `RUNNING` blocks `trigger-run` (HTTP 409); code-evaluation runs can take several minutes.
- The first run of a continuous task may show `CANCELLED` with no results (empty first window). Backfill with `trigger-run`.
- Sessions that start with `golden-exp-` come from experiments: leave them out of analysis (the task filters do not).
- An Arize "Signal" run on this project found no issues in the pilot, even with ten identical failures in two minutes: do not rely on it for this scenario.
