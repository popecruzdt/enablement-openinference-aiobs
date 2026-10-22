"""Telemetry helpers for custom OpenTelemetry instrumentation."""

from .context_attributes import ContextAttributesSpanProcessor
from .ollama_middle_span import instrument_ollama_middle_span
from .prompts import ANONYMOUS_USER_ID, prompt_template_context, request_context
from .setup import setup_tracing
from .workflows import workflow

__all__ = [
    "ANONYMOUS_USER_ID",
    "ContextAttributesSpanProcessor",
    "instrument_ollama_middle_span",
    "prompt_template_context",
    "request_context",
    "setup_tracing",
    "workflow",
]
