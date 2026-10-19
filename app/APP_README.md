# AI Travel Advisor Application

The easyTravel AI Travel Advisor app is a local-first FastAPI application that answers travel questions with Ollama and Weaviate. It serves a browser UI, exposes API endpoints for direct and retrieval-augmented responses, indexes destination HTML files into a vector store, and includes optional OpenTelemetry instrumentation for workshop observability scenarios.

This file is the source of truth for what the app does, how it is structured, and which runtime controls are currently supported.

## What the App Is

The application combines three pieces:

1. A FastAPI web server in `src/main.py` that serves the UI and API.
2. A retrieval pipeline in `src/rag/__init__.py` that loads destination HTML files, chunks them, creates embeddings through Ollama, and stores them in Weaviate.
3. A static front end in `public/index.html` that lets users switch between response modes and query the app interactively.

The app supports three response modes through `/api/v1/completion`:

- `rag`: recommended mode; retrieves indexed destination content and uses it as context.
- `llm`: direct model call without retrieval context.
- `agentic`: a LangGraph agent that uses tools (destination validation, KB search in Weaviate, current season) and answers from the tool results. See [Agentic workflow](#agentic-workflow).

## Runtime Architecture

At runtime the app is usually deployed with four services:

- `travel-advisor`: the FastAPI application container.
- `ollama`: local model runtime for chat and embeddings.
- `weaviate`: vector database used for the knowledge base.
- `nginx`: reverse proxy that publishes the UI in the compose flow.

An optional `otel-collector` service is also defined for OpenTelemetry export during observability demos.

High-level request flow:

1. The browser loads `/` from FastAPI directly or through Nginx.
2. The UI calls `/api/v1/completion?framework=...&prompt=...`.
3. In `llm` mode, the app sends the prompt directly to Ollama.
4. In `rag` mode, the app queries Weaviate for relevant destination chunks, builds a prompt from the retrieved context plus `prompts/rag_instructions.txt`, and then calls Ollama.
5. The response is returned as JSON and rendered in the UI.

Startup behavior matters:

- The app prepares the RAG knowledge base during FastAPI startup.
- It retries Weaviate initialization until the database is ready.
- If the knowledge base already has at least `MIN_KB_OBJECTS` objects and `FORCE_REINDEX` is not enabled, reindexing is skipped.

## Code Layout

Key files and directories:

- `src/main.py`: FastAPI app, startup lifecycle, API endpoints, and OpenTelemetry initialization.
- `src/config.py`: environment-backed settings loaded from `.env` and the process environment.
- `src/rag/__init__.py`: knowledge-base preparation, retrieval logic (`retrieve_context`, shared with the agent), prompt loading, and embedding drift simulation.
- `src/agent/`: LangGraph travel agent (`graph.py`) and its tools (`tools.py`).
- `src/feature_flags.py`: in-memory OpenFeature flag used to override the embedding model at runtime.
- `src/telemetry/`: custom Ollama span instrumentation and OpenInference-to-GenAI attribute mapping.
- `src/utils/__init__.py`: logging configuration and API response formatting.
- `public/index.html`: browser UI.
- `destinations/`: HTML source documents for the RAG knowledge base.
- `prompts/rag_instructions.txt`: prompt template used for RAG answers.
- `prompts/agentic_instructions.txt`: agent system prompt and final-answer prompt for agentic mode.
- `nginx/default.conf`: reverse proxy for the compose deployment.
- `Containerfile`: application image build.
- `podman-compose.yml`: local multi-container runtime definition.
- `opentelemetry/otel-collector-config.yaml`: collector config used by the optional compose collector.

## How It Works

### API surface

Primary endpoints:

- `GET /health`: liveness summary plus Ollama and Weaviate connectivity state.
- `GET /api/v1/status`: current model and retrieval settings.
- `GET /api/v1/completion`: response generation entry point.
- `GET /api/v1/prepare-kb`: rebuild the knowledge base from `destinations/`.
- `GET /api/v1/set-embedding-model`: set an in-memory embedding override used for drift simulation.
- `GET /api/v1/thumbsUp` and `GET /api/v1/thumbsDown`: lightweight feedback logging endpoints.

### Retrieval pipeline

The RAG flow in `src/rag/__init__.py` works like this:

1. Load every `*.html` file from `DESTINATIONS_PATH`.
2. Parse them with `BSHTMLLoader`.
3. Split content into chunks using `CHUNK_SIZE` and `CHUNK_OVERLAP`.
4. Create vectors in Weaviate using the Ollama embedding model named by `AI_EMBEDDING_MODEL`.
5. For a user query, try exact lexical retrieval first and then vector retrieval.
6. Build the final prompt from the retrieved text plus `RAG_PROMPT_PATH`.
7. Send that prompt to the chat model named by `AI_MODEL`.

If no useful context is found, the app returns a constrained fallback response instead of hallucinating details.

### Agentic workflow

Agentic mode runs a LangGraph `StateGraph` named `travel_agent` (`src/agent/graph.py`) with three tools (`src/agent/tools.py`):

- `validate_destination`: checks the destination against the KB destination list (the HTML file names in `DESTINATIONS_PATH`) and suggests close matches.
- `search_destination_kb`: retrieves destination context from Weaviate through the same `retrieve_context` path as RAG, so embedding drift affects agentic mode too.
- `get_current_season`: returns today's date and the season at the destination (northern, southern, or tropical). No external API is called.

The graph has two paths:

1. Tool calling: when the model in `AI_MODEL` has the Ollama `tools` capability (for example `qwen2.5` and `llama3.2`), the `call_model` node lets the model choose tools at temperature 0, and the `tools` node runs them in a loop of at most `AGENT_MAX_STEPS` model turns.
2. Fallback: when the model has no tool support (for example `gemma2`, `gemma3:1b`, `orca-mini`) or returns no tool calls, the `plan_fallback` node calls all three tools in a fixed order through the same `tools` node.

On both paths the `respond` node writes the final answer from the tool results using `AI_TEMPERATURE`. Tool support is detected once at startup with `ollama show`, so restart the app after changing `AI_MODEL`.

### Prompt control

The RAG response instructions are file-driven. The app loads `prompts/rag_instructions.txt` at runtime when the file exists, otherwise it falls back to an internal prompt string in `src/rag/__init__.py`.

The agentic prompts work the same way: `prompts/agentic_instructions.txt` holds a `### SYSTEM` section (tool-use instructions) and a `### RESPOND` section (final-answer template with `{tool_results}` and `{request}` placeholders), with fallbacks in `src/agent/graph.py`.

That means you can change response behavior without editing Python code by updating:

- `prompts/rag_instructions.txt`
- `prompts/agentic_instructions.txt`
- the destination HTML files in `destinations/`

After changing destination files, rebuild the knowledge base with `/api/v1/prepare-kb` or restart the app with `FORCE_REINDEX=true`.

## Runtime Control Surfaces

The app can be controlled in three main ways.

### 1. Environment variables

Settings are loaded from `.env` and the process environment. In containerized runs, service-level environment variables from `podman-compose.yml` override the image-baked `.env` file.

Important environment variables currently used by the app:

Core server:

- `SERVICE_NAME`
- `APP_VERSION`
- `ENVIRONMENT`
- `DEBUG`
- `HOST`
- `PORT`
- `LOG_LEVEL`

Ollama:

- `OLLAMA_ENDPOINT`
- `AI_MODEL`
- `AI_EMBEDDING_MODEL`
- `AI_TEMPERATURE`

Weaviate:

- `WEAVIATE_ENDPOINT`
- `WEAVIATE_PORT`
- `WEAVIATE_SCHEME`

RAG behavior:

- `MAX_PROMPT_LENGTH`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `FORCE_REINDEX`
- `MIN_KB_OBJECTS`

Agentic behavior:

- `AGENT_MAX_STEPS` (default `8`): maximum model turns in the tool-calling loop before the agent answers.

Filesystem paths:

- `DESTINATIONS_PATH`
- `PUBLIC_PATH`
- `RAG_PROMPT_PATH`
- `AGENTIC_PROMPT_PATH` (default `prompts/agentic_instructions.txt`)

OpenTelemetry and tracing:

- `OPENLLMETRY_ENABLED`
- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
- `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`
- `OTEL_SERVICE_NAME`
- `TRACELOOP_TRACE_CONTENT`

Instrumentation sources when `OPENLLMETRY_ENABLED=true`:

- LangChain and LangGraph spans (LLM calls, the `travel_agent` graph, its nodes and tools) come from OpenInference (`openinference-instrumentation-langchain`). Traceloop's LangChain instrumentor is blocked to avoid duplicate spans.
- Traceloop (OpenLLMetry) still sets up the tracer provider and exporter, the `@workflow` spans (`llm_advice_workflow`, `rag_advice_workflow`, `agentic_advice_workflow`), and its other instrumentors.
- Traceloop-created spans have no OpenInference kind of their own, so `_OpenInferenceKindSpanProcessor` sets `openinference.span.kind` when they start: `CHAIN` for `@workflow`/`@task` spans and Weaviate operations, `RETRIEVER` for Weaviate queries. Without it, Arize AX shows these spans with kind `unknown`. The tag reaches Arize through the OpenTelemetry export; OneAgent captures spans before it is set, so Dynatrace does not see it.
- OpenInference does not make its LangChain/LangGraph spans the current OpenTelemetry span, so spans created inside a tool or LLM call would be parented to the workflow span. `langchain_span_context()` (`src/telemetry/langchain_context.py`) activates the running tool or node span around KB retrieval (`RAGPipeline.retrieve_context`) and the `ollama.client.request` span, so the `weaviate.*` spans nest under `search_destination_kb`. If the span cannot be found, behavior falls back to the previous nesting. Spans captured natively by OneAgent (`weaviate.v1.Weaviate/*`, `POST`) are parented by OneAgent and stay under the request span.
- The OpenTelemetry HTTPX auto-instrumentation is not enabled. Weaviate queries use gRPC (covered by the gRPC client instrumentation), and HTTPX `POST` spans for Ollama `/api/chat` only duplicated the `ChatOllama` and `ollama.client.request` spans. As a result there are no OTel spans for the Weaviate startup REST calls or the embedding-drift `/api/embed` call. OneAgent still captures its own native `POST` span for Ollama calls.
- OpenInference spans are exported as emitted (OpenInference semantic conventions, Arize AX as the primary consumer). The resource attribute `openinference.project.name=ai-travel-advisor` sets the Arize AX project. The app does not add Dynatrace-specific attributes to these spans.
- For Dynatrace AI Observability, Dynatrace OpenPipeline maps OpenInference attributes to `gen_ai.*` at ingest, following [Dynatrace's OpenInference guidance](https://docs.dynatrace.com/docs/shortlink/ai-ml-openinference). Spans captured by OneAgent are routed by host group to `destination-automation-spans`; spans exported by the OpenTelemetry Collector (OpenTelemetry-only mode, no OneAgent) carry `service.namespace=destination-automation` and are routed to `destination-automation-otlp-spans`. Both pipelines are in the Monaco project under `dynatrace/config`.
- The OpenTelemetry Collector can also export traces to Arize AX (`otel_collector_arize_export`); see `ansible/deploy/docs/deploy.md`.

Note on legacy config values:

- `src/config.py` still defines some older telemetry-related settings such as `OTEL_ENABLED`, `OTEL_ENDPOINT`, `DYNATRACE_API_URL`, `DYNATRACE_API_TOKEN`, `API_TOKEN`, and `TRACELOOP_TELEMETRY`, but the current runtime instrumentation path is controlled by `OPENLLMETRY_ENABLED` and the `OTEL_EXPORTER_OTLP_*` variables above.

### 2. Runtime-managed files

These files directly change app behavior without code edits:

- `destinations/*.html`: the knowledge base source documents.
- `prompts/rag_instructions.txt`: the retrieval prompt template.
- `prompts/agentic_instructions.txt`: the agent and final-answer prompts for agentic mode.
- `public/index.html`: the browser UI.
- `nginx/default.conf`: external routing for the compose deployment.
- `opentelemetry/otel-collector-config.yaml`: collector export behavior in the compose deployment.

### 3. Control APIs

These endpoints change or refresh behavior at runtime:

- `/api/v1/prepare-kb`: rebuild indexed destination content.
- `/api/v1/set-embedding-model?model=...`: change the active embedding override in memory for drift simulation.

The embedding override is implemented with an in-process OpenFeature provider in `src/feature_flags.py`. It is not persisted across restarts.

Flag changes are observable. `src/telemetry/feature_flag_events.py` registers an OpenFeature `PROVIDER_CONFIGURATION_CHANGED` handler (only when `OPENLLMETRY_ENABLED=true`). When the flag value actually changes, it emits one OpenTelemetry log record over OTLP (event name `feature_flag.change`) with `feature_flag.key`, `feature_flag.provider.name`, `feature_flag.result.value` (empty string when the flag is at its default), `feature_flag.previous_value`, `feature_flag.result.variant` (the embedding model actually in use: the override, or the configured `AI_EMBEDDING_MODEL` at the default) and `feature_flag.result.enabled` (`true` while an override is active, `false` at the default). Setting the flag to its current value emits nothing. The record carries `service.name=ai-travel-advisor`; when OneAgent is detected, its entity IDs (process group instance, host, host group) are added to the log resource as well. The OpenTelemetry Collector exports it to Dynatrace (`otel_collector_dynatrace_export_logs`), where the destination-automation logs pipeline extracts a Davis event (`CUSTOM_CONFIGURATION`) from the record. The flag state is per app process, so the event is emitted by the worker that handled the change request.

## Local Run Modes

### Manual Python run

Use this when developing the app directly on the host:

```bash
cd /home/ec2-user/workshop-destination-automation/app
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m src.main
```

Manual runs typically expose the app on `http://localhost:8082` because `.env.example` sets `PORT=8082`.

### Podman Compose run

Use this when you want the full containerized stack defined in `podman-compose.yml`.

See `PODMAN_COMPOSE_SETUP.md` for the complete procedure.

Important compose-specific details:

- The app container listens on port `8080` internally.
- Nginx publishes the stack externally on `http://localhost:80`.
- The compose file mounts `src/`, `public/`, `prompts/`, and `destinations/` from the host into the app container.
- Changes to Python code and content files are visible inside the container immediately, but you still need to restart the app container for Python process changes to take effect because `DEBUG` is disabled in compose.

## Container Build Notes

The `Containerfile` builds a self-contained Python 3.13 image that:

- installs the Python dependencies from `requirements.txt`
- copies `src/`, `public/`, and `prompts/`
- copies `.env.example` to `.env` as the image default
- creates `/app/destinations`
- runs the process as a non-root user

Because the image bakes in `.env.example`, compose environment variables and bind mounts become the main override layers when running locally with Podman Compose.

## Common Operations

Query the app directly:

```bash
curl "http://localhost:8082/api/v1/completion?framework=rag&prompt=Paris"
curl "http://localhost:8082/api/v1/completion?framework=llm&prompt=Tokyo"
curl "http://localhost:8082/api/v1/completion?framework=agentic&prompt=Sydney"
```

Rebuild the knowledge base:

```bash
curl http://localhost:8082/api/v1/prepare-kb
```

Check health:

```bash
curl http://localhost:8082/health
curl http://localhost:8082/api/v1/status
```

When using the compose deployment through Nginx, replace `:8082` with `:80`.

## Troubleshooting

Common causes of startup or response issues:

- Ollama is up but the required models are not pulled.
- Weaviate is running but not yet ready when the app starts.
- `FORCE_REINDEX=true` triggers a full rebuild and makes startup slower.
- Destination HTML files changed but the KB was not rebuilt.
- The compose stack was started without the `travel-advisor` image being built first.

Useful checks:

```bash
curl http://localhost:8082/health
curl http://localhost:8082/api/v1/status
podman logs travel-advisor
podman logs ollama
podman logs weaviate
```

## Related Files

- `PODMAN_COMPOSE_SETUP.md`: step-by-step compose setup and lifecycle commands.
- `.env.example`: current example environment file for manual host-based runs.
