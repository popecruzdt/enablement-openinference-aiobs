"""`@workflow`: wrap a function in a CHAIN span named `<name>.workflow`.

The span records the call arguments as `input.value` and the result as `output.value`
and is the parent of the LangChain/LangGraph, Weaviate and Ollama spans created inside it.
With no tracer provider configured (OTEL_ENABLED is not true) the OpenTelemetry API
returns a no-op tracer, so the decorator is harmless.
"""

from __future__ import annotations

import functools
import inspect
import json
from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_TRACER_NAME = "ai-travel-advisor.workflow"
_SPAN_KIND_ATTRIBUTE = "openinference.span.kind"


def _to_text(value: Any) -> str:
    """Best-effort text for a span attribute. Never raises."""
    try:
        body = getattr(value, "body", None)  # fastapi/starlette responses
        if isinstance(body, (bytes, bytearray)):
            return bytes(body).decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str)
    except Exception:
        return repr(value)


def _record_input(span: trace.Span, args: tuple, kwargs: dict) -> None:
    try:
        if span.is_recording():
            span.set_attribute("input.value", _to_text({"args": list(args), "kwargs": kwargs}))
            span.set_attribute("input.mime_type", "application/json")
    except Exception:
        pass


def _record_output(span: trace.Span, result: Any) -> None:
    try:
        if span.is_recording():
            span.set_attribute("output.value", _to_text(result))
    except Exception:
        pass


def _record_error(span: trace.Span, exc: BaseException) -> None:
    try:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        pass


def workflow(name: Optional[str] = None) -> Callable:
    """Decorator factory: `@workflow(name="rag_advice_workflow")`."""

    def decorator(func: Callable) -> Callable:
        span_name = f"{name or func.__name__}.workflow"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = trace.get_tracer(_TRACER_NAME)
                with tracer.start_as_current_span(
                    span_name, attributes={_SPAN_KIND_ATTRIBUTE: "CHAIN"}, record_exception=False, set_status_on_exception=False
                ) as span:
                    _record_input(span, args, kwargs)
                    try:
                        result = await func(*args, **kwargs)
                    except BaseException as exc:
                        _record_error(span, exc)
                        raise
                    _record_output(span, result)
                    return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(_TRACER_NAME)
            with tracer.start_as_current_span(
                span_name, attributes={_SPAN_KIND_ATTRIBUTE: "CHAIN"}, record_exception=False, set_status_on_exception=False
            ) as span:
                _record_input(span, args, kwargs)
                try:
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    _record_error(span, exc)
                    raise
                _record_output(span, result)
                return result

        return wrapper

    return decorator
