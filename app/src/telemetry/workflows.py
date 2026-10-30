"""`@workflow`: wrap a function in a CHAIN span named `<name>.workflow`.

The span records the request as `input.value` and the answer as `output.value` (plain text when the
call has a single text argument and returns a response with a `message`; JSON otherwise)
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
    """`input.value` is the user's text for a call like `workflow("Atlanta")`, else the arguments as JSON."""
    try:
        if not span.is_recording():
            return
        if len(args) == 1 and not kwargs and isinstance(args[0], str):
            span.set_attribute("input.value", args[0])
            span.set_attribute("input.mime_type", "text/plain")
        else:
            span.set_attribute("input.value", _to_text({"args": list(args), "kwargs": kwargs}))
            span.set_attribute("input.mime_type", "application/json")
    except Exception:
        pass


def _record_output(span: trace.Span, result: Any) -> None:
    """`output.value` is the `message` of a response like `{"message": "...", "status": ...}`, else the whole result."""
    try:
        if not span.is_recording():
            return
        text = _to_text(result)
        mime_type = "text/plain"
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("message"), str):
                text = parsed["message"]
            else:
                mime_type = "application/json"
        span.set_attribute("output.value", text)
        span.set_attribute("output.mime_type", mime_type)
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
