"""
AI Travel Advisor Application
A modern FastAPI-based travel recommendation system using Ollama and Weaviate.
Powered by local LLMs with no external service dependencies.
"""

import asyncio
import contextvars
import os
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import ollama
from langchain_ollama.chat_models import ChatOllama
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
import uvicorn
from opentelemetry.sdk.trace import SpanProcessor

from src.config import get_settings
from src.utils import setup_logging, format_rag_response, format_error_response
from src.rag import get_rag_pipeline
from src.telemetry import instrument_ollama_middle_span
from src.telemetry.feature_flag_events import setup_feature_flag_events
from src.agent import (
    build_agent_graph,
    build_tools,
    load_agentic_prompts,
    load_known_destinations,
    model_supports_tools,
    run_agent,
)

# Configure logging
setup_logging()
logger = logging.getLogger("ai-travel-advisor")


# ==================== OpenTelemetry Support ====================
# Define no-op decorators by default (will be replaced if Traceloop is available and enabled)
def workflow(name=None):
    """No-op workflow decorator when Traceloop is disabled"""
    def decorator(func):
        return func
    return decorator

def task(name=None):
    """No-op task decorator when Traceloop is disabled"""
    def decorator(func):
        return func
    return decorator


_weaviate_instrumented = False
_opentelemetry_initialized = False

# Module-level metric instruments (populated after MeterProvider is initialized)
_inference_duration = None   # Histogram: gen_ai.client.operation.duration (seconds)
_inference_requests = None   # Counter:   gen_ai.client.inference.requests
_token_usage = None          # Histogram: gen_ai.client.token.usage (tokens)


class _GenAIModelSpanProcessor(SpanProcessor):
    """
    Enrich GenAI spans with all available common semantic attributes.
    
    Captures model names, provider info, operation names, token usage, and other metadata
    according to OpenTelemetry GenAI semantic conventions.
    References:
    - https://docs.dynatrace.com/docs/shortlink/genai-terms-and-concepts#common-attributes
    - https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def on_start(self, span, parent_context=None):
        """Set initial GenAI attributes at span start."""
        if span is None:
            return
        
        # Check if span is recording (safe check for both writable and read-only spans)
        try:
            if hasattr(span, "is_recording") and not span.is_recording():
                return
        except Exception:
            pass

        span_name = getattr(span, "name", "")
        # Process GenAI-related operations from OpenLLMetry/LangChain
        if _is_openinference_span(span) or not self._should_process_span(span_name):
            return

        try:
            # Set provider using standard attribute name
            span.set_attribute("gen_ai.provider.name", "ollama")
            # Also set the legacy attribute for backward compatibility
            span.set_attribute("gen_ai.system", "ollama")
            
            # Set model attributes to ensure correct values
            span.set_attribute("gen_ai.request.model", self.model_name)
            span.set_attribute("gen_ai.response.model", self.model_name)
            
            # Infer and set operation name if not already set by instrumentation
            current_op = _get_span_attribute(span, "gen_ai.operation.name")
            if current_op in (None, "", "unknown"):
                op_name = self._infer_operation_name(span_name)
                if op_name:
                    span.set_attribute("gen_ai.operation.name", op_name)
        except Exception:
            pass

    def on_end(self, span):
        """
        Enrich span with all available common GenAI attributes at completion.
        Ensures all standard attributes are present with valid, non-empty values.
        """
        if span is None:
            return

        # Check if span is recording (safe check for both writable and read-only spans)
        try:
            if hasattr(span, "is_recording") and not span.is_recording():
                return
        except Exception:
            pass

        span_name = getattr(span, "name", "")
        if _is_openinference_span(span) or not self._should_process_span(span_name):
            return

        try:
            # Ensure provider attributes are set with standard names
            if not _get_span_attribute(span, "gen_ai.provider.name"):
                _set_span_attribute_compat(span, "gen_ai.provider.name", "ollama")
            
            # Normalize model attributes to ensure they're never 'unknown'
            self._normalize_model_attributes(span)
            
            # Extract and ensure token usage attributes
            self._enrich_token_attributes(span)
            
            # Ensure operation name is set
            if not _get_span_attribute(span, "gen_ai.operation.name"):
                op_name = self._infer_operation_name(span_name)
                if op_name:
                    _set_span_attribute_compat(span, "gen_ai.operation.name", op_name)
                    
        except Exception:
            pass

    def _should_process_span(self, span_name: str) -> bool:
        """Check if this span should be processed as a GenAI operation."""
        if not span_name:
            return False
        # Match common GenAI span names from OpenLLMetry and LangChain
        genai_keywords = [
            "ChatOllama", "Ollama", "generate", "chat", "embedding",
            "invoke", "llm", "gen_ai", "langchain"
        ]
        return any(keyword.lower() in span_name.lower() for keyword in genai_keywords)

    def _infer_operation_name(self, span_name: str) -> str:
        """Infer operation type from span name."""
        span_lower = span_name.lower()
        if "chat" in span_lower:
            return "chat"
        elif "embedding" in span_lower or "embed" in span_lower:
            return "embeddings"
        elif "generate" in span_lower or "completion" in span_lower:
            return "generate"
        elif "agent" in span_lower or "invoke" in span_lower:
            return "invoke_agent"
        return "generate"  # default fallback

    def _normalize_model_attributes(self, span) -> None:
        """Ensure model attributes have correct, non-empty values."""
        req_model = _get_span_attribute(span, "gen_ai.request.model")
        if req_model in (None, "", "unknown"):
            _set_span_attribute_compat(span, "gen_ai.request.model", self.model_name)
        
        resp_model = _get_span_attribute(span, "gen_ai.response.model")
        if resp_model in (None, "", "unknown"):
            _set_span_attribute_compat(span, "gen_ai.response.model", self.model_name)

    def _enrich_token_attributes(self, span) -> None:
        """Extract and ensure token usage attributes are present."""
        # Try to extract input tokens from various possible attribute names
        input_tokens = self._extract_token_count(span, [
            "gen_ai.usage.input_tokens",
            "gen_ai.prompt_tokens",
            "llm.token.counts.prompt",
            "llm.usage.prompt_tokens",
        ])
        if input_tokens is not None:
            _set_span_attribute_compat(span, "gen_ai.usage.input_tokens", str(input_tokens))
        
        # Try to extract output tokens from various possible attribute names
        output_tokens = self._extract_token_count(span, [
            "gen_ai.usage.output_tokens",
            "gen_ai.completion_tokens",
            "llm.token.counts.completion",
            "llm.usage.completion_tokens",
        ])
        if output_tokens is not None:
            _set_span_attribute_compat(span, "gen_ai.usage.output_tokens", str(output_tokens))

    def _extract_token_count(self, span, attribute_keys: list) -> Optional[int]:
        """Try to extract token count from span attributes using multiple keys."""
        try:
            for key in attribute_keys:
                value = _get_span_attribute(span, key)
                if value is not None:
                    if isinstance(value, str):
                        return int(value)
                    elif isinstance(value, (int, float)):
                        return int(value)
        except (ValueError, TypeError):
            pass
        return None

    def shutdown(self):
        return

    def force_flush(self, timeout_millis=30000):
        return True


class _OpenInferenceKindSpanProcessor(SpanProcessor):
    """Tag Traceloop-created spans with `openinference.span.kind`.

    Arize AX classifies spans by `openinference.span.kind`; Traceloop's @workflow/@task
    spans and Weaviate spans only carry `traceloop.span.kind`, so they show as "unknown".
    The kind is set in on_start (while the span is open) so it is exported as emitted.
    Spans that already have a kind (OpenInference instrumentation) are left alone.
    """

    _RETRIEVER_MARKERS = (".collections.query.", ".gql.query.get", ".gql.get")

    def on_start(self, span, parent_context=None):
        try:
            if span is None or _is_openinference_span(span):
                return
            kind = self._kind_for(getattr(span, "name", "") or "")
            if kind and not _get_span_attribute(span, "openinference.span.kind"):
                span.set_attribute("openinference.span.kind", kind)
        except Exception:
            # Tracing must never affect request handling.
            pass

    def _kind_for(self, span_name: str) -> Optional[str]:
        if span_name.endswith((".workflow", ".task")):
            return "CHAIN"
        if span_name.startswith(("db.weaviate.", "weaviate.")):
            if any(marker in span_name for marker in self._RETRIEVER_MARKERS):
                return "RETRIEVER"
            return "CHAIN"
        return None

    def on_end(self, span):
        return

    def shutdown(self):
        return

    def force_flush(self, timeout_millis=30000):
        return True


def _is_openinference_span(span) -> bool:
    """True for spans from OpenInference instrumentation.

    OpenInference spans are left as emitted (for Arize AX); Dynatrace gen_ai.* mapping
    for them is done in OpenPipeline, not in the app.
    """
    scope = getattr(span, "instrumentation_scope", None) or getattr(span, "instrumentation_info", None)
    return str(getattr(scope, "name", "") or "").startswith("openinference")


def _get_span_attribute(span, key: str):
    """Read span attribute from both mutable and readable span representations."""
    try:
        attrs = getattr(span, "attributes", None)
        if attrs is not None:
            value = attrs.get(key)
            if value is not None:
                return value
    except Exception:
        pass

    try:
        attrs = getattr(span, "_attributes", None)
        if attrs is not None:
            value = attrs.get(key)
            if value is not None:
                return value
    except Exception:
        pass

    return None


def _set_span_attribute_compat(span, key: str, value: str) -> None:
    """Set span attribute across writable and readable span types."""
    try:
        if hasattr(span, "set_attribute"):
            span.set_attribute(key, value)
            return
    except Exception:
        pass

    try:
        attrs = getattr(span, "attributes", None)
        if attrs is not None:
            attrs[key] = value
            return
    except Exception:
        pass

    try:
        attrs = getattr(span, "_attributes", None)
        if attrs is not None:
            attrs[key] = value
    except Exception:
        pass


def _normalize_genai_model_attributes(span, fallback_model_name: str) -> None:
    """Replace unknown GenAI model attributes on spans before export."""
    try:
        provider_model = _get_span_attribute(span, "traceloop.association.properties.ls_model_name")
        model_name = provider_model or fallback_model_name
        if not model_name:
            return

        req_model = _get_span_attribute(span, "gen_ai.request.model")
        resp_model = _get_span_attribute(span, "gen_ai.response.model")

        if req_model in (None, "", "unknown"):
            _set_span_attribute_compat(span, "gen_ai.request.model", model_name)
        if resp_model in (None, "", "unknown"):
            _set_span_attribute_compat(span, "gen_ai.response.model", model_name)
        if _get_span_attribute(span, "gen_ai.provider.name") in (None, ""):
            _set_span_attribute_compat(span, "gen_ai.provider.name", "ollama")
        if _get_span_attribute(span, "gen_ai.system") in (None, ""):
            _set_span_attribute_compat(span, "gen_ai.system", "ollama")
    except Exception:
        # Never fail tracing/export because of attribute normalization.
        pass


def _set_genai_request_attributes(model_name: str) -> None:
    """Attach GenAI semantic attributes to the current span when available."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.response.model", model_name)
            span.set_attribute("gen_ai.system", "ollama")
    except Exception:
        # Tracing must never affect request handling.
        pass


def setup_openinference_instrumentation():
    """Instrument LangChain/LangGraph with OpenInference on the global tracer provider."""
    try:
        from opentelemetry import trace
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=trace.get_tracer_provider())
        logger.info("Enabled OpenInference instrumentation for LangChain/LangGraph")
    except Exception as e:
        logger.warning(f"Could not enable OpenInference instrumentation for LangChain: {e}")


def setup_weaviate_instrumentation():
    """Instrument Weaviate client gRPC traffic with OpenTelemetry.

    HTTPX is deliberately not instrumented: Weaviate queries use gRPC, and the HTTPX spans only
    duplicated the Ollama chat calls already covered by the OpenInference and ollama.client.request spans.
    """
    global _weaviate_instrumented

    if _weaviate_instrumented:
        return

    # Some Weaviate operations can use gRPC. Enable client instrumentation if installed.
    try:
        from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

        GrpcInstrumentorClient().instrument()
        logger.info("Enabled gRPC instrumentation for Weaviate operations")
    except ImportError:
        logger.info("gRPC instrumentation package not installed; skipping Weaviate gRPC tracing")
    except Exception as e:
        logger.warning(f"Could not enable gRPC instrumentation for Weaviate: {e}")

    _weaviate_instrumented = True


def _initialize_otlp_metrics(otlp_endpoint: str) -> None:
    """Set up an OTLP metrics pipeline and register global metric instruments.

    The MeterProvider is created directly and instruments are obtained from it
    without relying on the global set_meter_provider() call, which may have
    already been locked by Traceloop SDK.
    """
    global _inference_duration, _inference_requests, _token_usage
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource

        metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", otlp_endpoint)
        insecure = metrics_endpoint.startswith("http://")
        metric_exporter = OTLPMetricExporter(endpoint=metrics_endpoint, insecure=insecure)
        reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
        resource = Resource.create({"service.name": "ai-travel-advisor"})
        # Create provider directly — do NOT call set_meter_provider() as Traceloop
        # may have already locked the global provider slot (it prints "Metrics are disabled").
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])

        meter = meter_provider.get_meter("ai-travel-advisor", "1.0.0")
        _inference_duration = meter.create_histogram(
            name="gen_ai.client.operation.duration",
            unit="s",
            description="Duration of GenAI inference operations in seconds",
        )
        _inference_requests = meter.create_counter(
            name="gen_ai.client.inference.requests",
            unit="{request}",
            description="Total number of GenAI inference requests",
        )
        _token_usage = meter.create_histogram(
            name="gen_ai.client.token.usage",
            unit="{token}",
            description="Token usage per GenAI inference request",
        )
        logger.info(f"OTel metrics pipeline initialized with endpoint: {metrics_endpoint}")
    except Exception as exc:
        logger.warning(f"Could not initialize OTel metrics pipeline: {exc}")


def initialize_opentelemetry():
    """Initialize OpenTelemetry instrumentation if enabled"""
    global workflow, task, _opentelemetry_initialized

    if _opentelemetry_initialized:
        return
    
    openllmetry_enabled = os.getenv("OPENLLMETRY_ENABLED", "false").lower() == "true"
    
    if not openllmetry_enabled:
        logger.info("OpenLLMetry instrumentation is disabled (OPENLLMETRY_ENABLED is not set to true)")
        return
    
    try:
        from traceloop.sdk import Traceloop
        from traceloop.sdk.decorators import workflow as workflow_decorator, task as task_decorator
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        
        workflow = workflow_decorator
        task = task_decorator
        
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        )
        insecure = otlp_endpoint.startswith("http://")
        logger.info(f"Initializing OpenLLMetry instrumentation with endpoint: {otlp_endpoint}")

        # Use explicit OTLP exporter so traces are sent to the local collector.
        trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=insecure)

        # Traceloop API changed across versions; support both variants.
        if hasattr(Traceloop, "init"):
            from traceloop.sdk.instruments import Instruments

            ai_model = get_settings().ai_model
            default_processor = Traceloop.get_default_span_processor(
                exporter=trace_exporter,
            )
            model_attr_processor = _GenAIModelSpanProcessor(ai_model)
            oi_kind_processor = _OpenInferenceKindSpanProcessor()
            # LangChain/LangGraph spans come from OpenInference (below), so Traceloop's
            # LangChain instrumentor is blocked to avoid a duplicate span tree.
            # openinference.project.name groups traces into a project in Arize AX.
            Traceloop.init(
                app_name="ai-travel-advisor",
                processor=[default_processor, model_attr_processor, oi_kind_processor],
                block_instruments={Instruments.LANGCHAIN},
                resource_attributes={"openinference.project.name": "ai-travel-advisor"},
                span_postprocess_callback=lambda span: _normalize_genai_model_attributes(
                    span,
                    ai_model,
                ),
            )

        else:
            Traceloop.initialize(
                app_name="ai-travel-advisor",
                exporter_url=otlp_endpoint,
            )

        setup_openinference_instrumentation()
        setup_feature_flag_events(otlp_endpoint, insecure)
        setup_weaviate_instrumentation()
        instrument_ollama_middle_span()
        _initialize_otlp_metrics(otlp_endpoint)
        _opentelemetry_initialized = True

        logger.info("OpenLLMetry instrumentation initialized successfully")
    except ImportError:
        logger.warning("Traceloop SDK not installed. Install with: pip install traceloop-sdk")
        logger.warning("Continuing without OpenLLMetry instrumentation")
    except Exception as e:
        logger.error(f"Failed to initialize OpenLLMetry: {e}")
        logger.warning("Continuing without OpenLLMetry instrumentation")


# Initialize OpenTelemetry during module import so decorators below bind correctly.
# If disabled or initialization fails, the no-op decorators remain in place.
initialize_opentelemetry()


# ==================== Application State ====================
class AppState:
    """Global application state"""
    rag_pipeline = None
    ollama_client = None
    direct_chat_model = None
    agent_graph = None
    settings = None


async def prepare_rag_pipeline_on_startup(max_attempts: int = 18, retry_delay: int = 5) -> None:
    """Prepare the KB during startup so RAG is ready before serving requests."""
    if AppState.rag_pipeline is None:
        AppState.rag_pipeline = get_rag_pipeline()

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"Preparing RAG pipeline on startup (attempt {attempt}/{max_attempts})")
            AppState.rag_pipeline.connect_weaviate()
            AppState.rag_pipeline.prepare_knowledge_base()
            AppState.rag_pipeline.initialize_rag_chain()
            logger.info("RAG pipeline prepared during startup")
            return
        except Exception as e:
            last_error = e
            logger.warning(f"Startup RAG preparation attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(retry_delay)

    raise RuntimeError(f"Failed to prepare RAG pipeline during startup: {last_error}")


async def initialize_app():
    """Initialize application resources"""
    logger.info("Initializing AI Travel Advisor...")

    settings = get_settings()
    AppState.settings = settings
    
    # Verify Ollama connection
    try:
        AppState.ollama_client = ollama.Client(host=settings.ollama_endpoint)
        models = AppState.ollama_client.list()
        logger.info(f"Connected to Ollama at {settings.ollama_endpoint}")

        # ollama-python response shape can vary by version (dict vs typed object).
        model_names = []
        if isinstance(models, dict):
            model_names = [item.get("name", "") for item in models.get("models", []) if item.get("name")]
        elif hasattr(models, "models"):
            model_names = [getattr(item, "model", "") for item in models.models if getattr(item, "model", "")]

        logger.info(f"Available models: {model_names}")

        # Initialize Direct LLM path with LangChain so span structure/attributes
        # match the RAG workflow (ChatOllama.chat with same GenAI metadata flow).
        AppState.direct_chat_model = ChatOllama(
            model=settings.ai_model,
            base_url=settings.ollama_endpoint,
            temperature=settings.ai_temperature,
        )
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}")
        logger.info("Note: Make sure Ollama is running and accessible")
    
    # Initialize RAG pipeline
    try:
        await prepare_rag_pipeline_on_startup()
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.error(f"RAG pipeline initialization failed: {e}")
        raise

    # Initialize agentic workflow (requires Ollama and the RAG pipeline)
    if AppState.ollama_client and AppState.direct_chat_model:
        try:
            initialize_agent_graph(settings)
        except Exception as e:
            logger.warning(f"Agentic workflow initialization failed: {e}")

    logger.info("AI Travel Advisor initialized successfully")


def initialize_agent_graph(settings) -> None:
    """Build the LangGraph travel agent with tools bound to the shared RAG pipeline."""
    known_destinations = load_known_destinations(settings.destinations_path)
    tools = build_tools(AppState.rag_pipeline, known_destinations)
    system_prompt, respond_prompt = load_agentic_prompts(settings.agentic_prompt_path)
    supports_tools = model_supports_tools(AppState.ollama_client, settings.ai_model)

    AppState.agent_graph = build_agent_graph(
        chat_model=AppState.direct_chat_model,
        tools=tools,
        known_destinations=known_destinations,
        system_prompt=system_prompt,
        respond_prompt=respond_prompt,
        supports_tools=supports_tools,
        max_steps=settings.agent_max_steps,
    )
    path = "tool calling" if supports_tools else "fallback tool plan"
    logger.info(f"Agentic workflow initialized with {len(tools)} tools ({path})")


async def shutdown_app():
    """Cleanup application resources"""
    logger.info("Shutting down AI Travel Advisor...")
    if AppState.rag_pipeline and AppState.rag_pipeline.weaviate_client:
        try:
            AppState.rag_pipeline.weaviate_client.close()
        except Exception as e:
            logger.warning(f"Error closing Weaviate connection: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup, clean up on shutdown"""
    await initialize_app()
    yield
    await shutdown_app()


# ==================== FastAPI Application ====================
app = FastAPI(
    title="AI Travel Advisor",
    description="Local-first AI travel recommendations using Ollama and Weaviate",
    version="1.0.0",
    lifespan=lifespan,
)


# ==================== Middleware ====================
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests"""
    logger.debug(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    return response


# ==================== Health & Status Endpoints ====================
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status and connected service information.
    """
    settings = get_settings()
    status_info = {
        "status": "healthy",
        "service": "AI Travel Advisor",
        "version": "1.0.0",
        "ollama": {
            "endpoint": settings.ollama_endpoint,
            "model": settings.ai_model,
            "connected": AppState.ollama_client is not None
        },
        "weaviate": {
            "endpoint": f"{settings.weaviate_scheme}://{settings.weaviate_endpoint}:{settings.weaviate_port}",
            "connected": AppState.rag_pipeline is not None and AppState.rag_pipeline.weaviate_client is not None
        }
    }
    return JSONResponse(content=status_info)


@app.get("/api/v1/status", tags=["Health"])
async def status():
    """Get detailed application status"""
    settings = get_settings()
    return JSONResponse(content={
        "service": "AI Travel Advisor",
        "status": "operational",
        "configuration": {
            "ollama_model": settings.ai_model,
            "ollama_temperature": settings.ai_temperature,
            "embedding_model": settings.ai_embedding_model,
            "chunk_size": settings.chunk_size,
            "retrieval_k": settings.retrieval_k
        }
    })


# ==================== Travel Advisor Endpoints ====================
@app.get("/api/v1/completion", tags=["Travel Advisor"])
async def submit_completion(framework: str = "rag", prompt: str = "") -> JSONResponse:
    """
    Get travel advice using different frameworks.
    
    Args:
        framework: "llm" (direct LLM), "rag" (retrieval augmented), or "agentic"
        prompt: The destination or query
        
    Returns:
        Travel advice response
    """
    if not prompt or len(prompt) > get_settings().max_prompt_length:
        return JSONResponse(
            status_code=400,
            content=format_error_response(
                f"Prompt must be provided and under {get_settings().max_prompt_length} characters",
                400
            )
        )
    
    try:
        if framework == "llm":
            return await llm_advice(prompt)
        elif framework == "rag":
            return await rag_advice(prompt)
        elif framework == "agentic":
            return await agentic_advice(prompt)
        else:
            return JSONResponse(
                status_code=400,
                content=format_error_response(f"Framework '{framework}' not supported. Use 'llm', 'rag', or 'agentic'.", 400)
            )
    except Exception as e:
        logger.error(f"Error in completion endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content=format_error_response(str(e), 500)
        )


@workflow(name="llm_advice_workflow")
async def llm_advice(destination: str) -> JSONResponse:
    """Get travel advice using a direct LangChain ChatOllama call."""
    logger.info(f"Getting LLM advice for: {destination}")
    
    if not AppState.ollama_client:
        return JSONResponse(
            status_code=503,
            content=format_error_response("Ollama not connected", 503)
        )
    
    try:
        prompt = f"Give travel advice in a paragraph of max 50 words about {destination}"
        _set_genai_request_attributes(get_settings().ai_model)

        if AppState.direct_chat_model is None:
            settings = get_settings()
            AppState.direct_chat_model = ChatOllama(
                model=settings.ai_model,
                base_url=settings.ollama_endpoint,
                temperature=settings.ai_temperature,
            )

        _t0 = time.monotonic()
        # Offload synchronous LangChain/Ollama call to a worker thread so
        # the event loop can continue serving health checks and other requests.
        current_context = contextvars.copy_context()
        response = await asyncio.to_thread(
            lambda: current_context.run(AppState.direct_chat_model.invoke, prompt)
        )
        _elapsed = time.monotonic() - _t0
        advice = response.content if hasattr(response, "content") else str(response)

        _attrs = {
            "gen_ai.system": "ollama",
            "gen_ai.request.model": get_settings().ai_model,
            "gen_ai.operation.name": "chat",
            "framework": "llm",
        }
        if _inference_duration:
            _inference_duration.record(_elapsed, attributes=_attrs)
        if _inference_requests:
            _inference_requests.add(1, attributes=_attrs)
        if _token_usage:
            meta = getattr(response, "response_metadata", {}) or {}
            input_tokens = meta.get("prompt_eval_count")
            output_tokens = meta.get("eval_count")
            if input_tokens:
                _token_usage.record(input_tokens, attributes={**_attrs, "gen_ai.token.type": "input"})
            if output_tokens:
                _token_usage.record(output_tokens, attributes={**_attrs, "gen_ai.token.type": "output"})

        return JSONResponse(content=format_rag_response(advice))
    except Exception as e:
        logger.error(f"LLM advice error: {e}")
        if _inference_requests:
            _inference_requests.add(1, attributes={
                "gen_ai.system": "ollama",
                "gen_ai.request.model": get_settings().ai_model,
                "gen_ai.operation.name": "chat",
                "framework": "llm",
                "error.type": type(e).__name__,
            })
        return JSONResponse(
            status_code=500,
            content=format_error_response(str(e), 500)
        )


@workflow(name="rag_advice_workflow")
async def rag_advice(destination: str) -> JSONResponse:
    """Get travel advice using RAG pipeline"""
    logger.info(f"Getting RAG advice for: {destination}")
    
    if not AppState.rag_pipeline:
        return JSONResponse(
            status_code=503,
            content=format_error_response("RAG pipeline not initialized", 503)
        )

    if not AppState.rag_pipeline.rag_chain:
        return JSONResponse(
            status_code=503,
            content=format_error_response("RAG pipeline is not ready yet", 503)
        )
    
    try:
        _t0 = time.monotonic()
        # RAG retrieval + generation is synchronous and potentially long-running.
        # Run it in a worker thread to avoid blocking the async event loop.
        current_context = contextvars.copy_context()
        advice = await asyncio.to_thread(
            lambda: current_context.run(AppState.rag_pipeline.get_travel_advice, destination)
        )
        _elapsed = time.monotonic() - _t0

        _attrs = {
            "gen_ai.system": "ollama",
            "gen_ai.request.model": get_settings().ai_model,
            "gen_ai.operation.name": "chat",
            "framework": "rag",
        }
        if _inference_duration:
            _inference_duration.record(_elapsed, attributes=_attrs)
        if _inference_requests:
            _inference_requests.add(1, attributes=_attrs)

        return JSONResponse(content=format_rag_response(advice))
    except Exception as e:
        logger.error(f"RAG advice error: {e}")
        if _inference_requests:
            _inference_requests.add(1, attributes={
                "gen_ai.system": "ollama",
                "gen_ai.request.model": get_settings().ai_model,
                "gen_ai.operation.name": "chat",
                "framework": "rag",
                "error.type": type(e).__name__,
            })
        return JSONResponse(
            status_code=500,
            content=format_error_response(str(e), 500)
        )


@workflow(name="agentic_advice_workflow")
async def agentic_advice(destination: str) -> JSONResponse:
    """Get travel advice using the LangGraph travel agent and its tools"""
    logger.info(f"Getting agentic advice for: {destination}")

    if not AppState.ollama_client or not AppState.rag_pipeline:
        return JSONResponse(
            status_code=503,
            content=format_error_response("Ollama or RAG pipeline not available", 503)
        )

    if not AppState.agent_graph:
        return JSONResponse(
            status_code=503,
            content=format_error_response("Agentic workflow is not ready yet", 503)
        )

    _attrs = {
        "gen_ai.system": "ollama",
        "gen_ai.request.model": get_settings().ai_model,
        "gen_ai.operation.name": "invoke_agent",
        "framework": "agentic",
    }
    try:
        _t0 = time.monotonic()
        # The agent graph runs synchronous LangChain/Ollama/Weaviate calls.
        # Run it in a worker thread to avoid blocking the async event loop.
        current_context = contextvars.copy_context()
        result = await asyncio.to_thread(
            lambda: current_context.run(
                run_agent, AppState.agent_graph, destination, get_settings().agent_max_steps
            )
        )
        _elapsed = time.monotonic() - _t0
        logger.info(f"Agentic workflow completed using the {result.get('path')} path")

        if _inference_duration:
            _inference_duration.record(_elapsed, attributes=_attrs)
        if _inference_requests:
            _inference_requests.add(1, attributes=_attrs)

        return JSONResponse(content=format_rag_response(result.get("answer", "")))
    except Exception as e:
        logger.error(f"Agentic advice error: {e}")
        if _inference_requests:
            _inference_requests.add(1, attributes={**_attrs, "error.type": type(e).__name__})
        return JSONResponse(
            status_code=500,
            content=format_error_response(str(e), 500)
        )


@workflow(name="prepare_knowledge_base_workflow")
@app.get("/api/v1/prepare-kb", tags=["Knowledge Base"])
async def prepare_knowledge_base() -> JSONResponse:
    """
    Rebuild the knowledge base with destination documents.
    This endpoint reloads and reindexes all HTML files from the destinations directory.
    """
    logger.info("Starting knowledge base preparation...")
    
    if not AppState.rag_pipeline:
        return JSONResponse(
            status_code=503,
            content=format_error_response("RAG pipeline not initialized", 503)
        )
    
    try:
        AppState.rag_pipeline.prepare_knowledge_base()
        AppState.rag_pipeline.initialize_rag_chain()
        return JSONResponse(content={
            "status": "success",
            "message": "Knowledge base prepared successfully"
        })
    except Exception as e:
        logger.error(f"Knowledge base preparation error: {e}")
        return JSONResponse(
            status_code=500,
            content=format_error_response(f"Knowledge base preparation failed: {str(e)}", 500)
        )


@app.get("/api/v1/set-embedding-model", tags=["Knowledge Base"])
async def set_embedding_model(model: str) -> JSONResponse:
    """
    Set or clear the embedding model override feature flag.
    When overridden to a mismatched model (e.g. gemma2:2b), near_text queries produce a
    vector dimension mismatch against stored 768-dim vectors, simulating embedding model drift.
    Distance metrics spike to 1.0 and result count drops to 0, triggering a Dynatrace anomaly.
    To restore: call with the configured AI_EMBEDDING_MODEL value (e.g. nomic-embed-text).
    """
    if not model:
        return JSONResponse(
            status_code=400,
            content=format_error_response("model parameter is required", 400)
        )

    if not AppState.rag_pipeline:
        return JSONResponse(
            status_code=503,
            content=format_error_response("RAG pipeline not initialized", 503)
        )

    try:
        result = AppState.rag_pipeline.set_embedding_model(model)
        logger.info(f"Embedding model override set via API: {result}")
        return JSONResponse(content={"status": "success", **result})
    except Exception as e:
        logger.error(f"Failed to set embedding model override: {e}")
        return JSONResponse(
            status_code=500,
            content=format_error_response(f"Failed to set embedding model override: {str(e)}", 500)
        )


@app.get("/api/v1/thumbsUp", tags=["Feedback"])
async def thumbs_up(prompt: str = "") -> JSONResponse:
    """Log positive user feedback for a search term"""
    logger.info(f"Positive feedback for search term: {prompt}")
    return JSONResponse(content={"status": "logged", "type": "positive"})


@app.get("/api/v1/thumbsDown", tags=["Feedback"])
async def thumbs_down(prompt: str = "") -> JSONResponse:
    """Log negative user feedback for a search term"""
    logger.info(f"Negative feedback for search term: {prompt}")
    return JSONResponse(content={"status": "logged", "type": "negative"})


# ==================== Static Files ====================
def setup_static_files():
    """Mount static files for the web UI"""
    public_path = get_settings().public_path
    if os.path.exists(public_path):
        app.mount("/", StaticFiles(directory=public_path, html=True), name="public")
        logger.info(f"Static files mounted from {public_path}")
    else:
        logger.warning(f"Public directory not found at {public_path}")


# Mount static files during module import so uvicorn `src.main:app` serves UI.
setup_static_files()


# ==================== Application Entry Point ====================
if __name__ == "__main__":
    settings = get_settings()
    
    logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
    logger.info(f"OpenAPI documentation available at http://{settings.host}:{settings.port}/docs")
    
    app_target = "src.main:app" if settings.debug else app

    uvicorn.run(
        app_target,
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
