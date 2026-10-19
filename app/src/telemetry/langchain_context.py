"""Make the active LangChain/LangGraph OpenInference span the current OpenTelemetry span.

OpenInference's LangChain tracer starts a span per run but deliberately never attaches it as the
current span. Spans created by app code inside a tool or LLM call (Weaviate, Ollama client) would
otherwise be parented to whatever span is current (the Traceloop workflow span) instead of the
tool or LLM span they run in.

`langchain_span_context()` finds the span of the LangChain run that is executing and activates it
for the duration of the block. It never ends the span and never raises: if the span cannot be
found (no LangChain run, or a LangChain/OpenInference change), the block runs unchanged.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from opentelemetry import trace

logger = logging.getLogger("ai-travel-advisor")


def _current_langchain_span():
    """Return the OpenInference span of the running LangChain run, or None."""
    from langchain_core.runnables.config import ensure_config

    callbacks = ensure_config().get("callbacks")
    run_id = getattr(callbacks, "parent_run_id", None)
    if run_id is None:
        return None
    for handler in getattr(callbacks, "handlers", None) or []:
        get_span = getattr(handler, "get_span", None)
        if callable(get_span):
            span = get_span(run_id)
            if span is not None:
                return span
    return None


@contextmanager
def langchain_span_context():
    """Run the block with the running LangChain run's span as the current span, if there is one."""
    try:
        span = _current_langchain_span()
    except Exception:
        logger.debug("Could not resolve the current LangChain span", exc_info=True)
        span = None
    if span is None:
        yield
        return
    with trace.use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
        yield
