"""Custom Ollama client instrumentation.

Creates a middle custom span wrapping the actual HTTP request so the hierarchy becomes:
ChatOllama.chat -> ollama.client.request (wraps HTTP POST) -> POST.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any
from urllib.parse import urljoin

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from .langchain_context import langchain_span_context

logger = logging.getLogger("ai-travel-advisor")
_tracer = trace.get_tracer("ai-travel-advisor.ollama")
_instrumented = False


def instrument_ollama_middle_span() -> None:
    """Patch Ollama client methods to emit a middle custom span.
    
    Patches the internal _stream method which is called for ALL requests,
    ensuring we capture the full response including timing metadata before
    closing the middle span.
    """
    global _instrumented

    if _instrumented:
        return

    try:
        import ollama
    except Exception as exc:
        logger.warning(f"Could not import ollama for custom span instrumentation: {exc}")
        return

    patched = 0

    # Patch sync and async client _stream methods (lower layer - actual HTTP requests)
    for cls_name in ("Client", "AsyncClient"):
        cls = getattr(ollama, cls_name, None)
        if cls is None:
            continue

        if _patch_stream_method(cls):
            patched += 1

    if patched > 0:
        _instrumented = True
        logger.info("Enabled custom Ollama middle-span instrumentation")
    else:
        logger.warning("No Ollama methods patched for middle-span instrumentation")


def _patch_stream_method(client_cls: type) -> bool:
    """Patch Ollama _stream method which handles all HTTP requests.
    
    The _stream method is the internal layer that actually sends HTTP requests
    and receives responses. By patching here instead of at chat/generate level,
    we ensure the full response (including timing metadata) is captured before
    the middle span closes.
    """
    # Check if this client class has the _stream method
    original = getattr(client_cls, "_stream", None)
    if original is None:
        return False
    
    # Check for idempotency
    if getattr(original, "_ai_travel_advisor_ollama_patched", False):
        return False
    
    if inspect.iscoroutinefunction(original):
        wrapped = _build_enhanced_async_stream_wrapper(original)
    else:
        wrapped = _build_enhanced_sync_stream_wrapper(original)
    
    wrapped._ai_travel_advisor_ollama_patched = True  # type: ignore[attr-defined]
    setattr(client_cls, "_stream", wrapped)
    logger.debug(f"Patched {client_cls.__name__}._stream for instrumentation")
    
    return True


def _build_enhanced_sync_stream_wrapper(original: Any):
    """Sync wrapper for Ollama _stream that captures the full response including timing."""
    @functools.wraps(original)
    def wrapped(self, method: str, url: str, *args, **kwargs):
        with langchain_span_context(), _tracer.start_as_current_span(
            "ollama.client.request", 
            kind=SpanKind.INTERNAL
        ) as span:
            # Set request attributes
            span.set_attribute("gen_ai.system", "ollama")
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", url)
            
            # Try to extract model from request body if present
            _set_request_model_attribute(span, kwargs)
            
            try:
                # Call original _stream which returns a response stream/iterator
                response_iter = original(self, method, url, *args, **kwargs)
                
                # Fully consume the response to ensure complete transmission
                # This is crucial: the timing metadata comes in the response body
                full_response = _consume_response_stream(response_iter)
                
                span.set_attribute("http.status_code", 200)
                
                # Extract response attributes from the full response
                _set_response_attributes_from_stream(span, full_response)
                
                # Yield/return the full response
                try:
                    if hasattr(full_response, '__iter__') and not isinstance(full_response, (str, bytes)):
                        yield from full_response
                    else:
                        yield full_response
                except TypeError:
                    # full_response might not be iterable
                    if full_response is not None:
                        yield full_response
                
            except Exception as exc:
                logger.exception(f"Ollama _stream instrumentation error: {exc}")
                _set_exception_attributes(span, exc)
                raise
    
    return wrapped


def _build_enhanced_async_stream_wrapper(original: Any):
    """Async wrapper for Ollama _stream that captures the full response including timing."""
    @functools.wraps(original)
    async def wrapped(self, method: str, url: str, *args, **kwargs):
        with langchain_span_context(), _tracer.start_as_current_span(
            "ollama.client.request",
            kind=SpanKind.INTERNAL
        ) as span:
            # Set request attributes
            span.set_attribute("gen_ai.system", "ollama")
            span.set_attribute("http.method", method)
            span.set_attribute("http.url", url)
            
            # Try to extract model from request body if present
            _set_request_model_attribute(span, kwargs)
            
            try:
                # Call original _stream which returns an async generator
                response_iter = original(self, method, url, *args, **kwargs)
                
                # Fully consume the response to ensure complete transmission
                full_response = await _consume_async_response_stream(response_iter)
                
                span.set_attribute("http.status_code", 200)
                
                # Extract response attributes from the full response
                _set_response_attributes_from_stream(span, full_response)
                
                # Yield the full response
                try:
                    if hasattr(full_response, '__aiter__'):
                        async for item in full_response:
                            yield item
                    elif hasattr(full_response, '__iter__') and not isinstance(full_response, (str, bytes)):
                        for item in full_response:
                            yield item
                    else:
                        if full_response is not None:
                            yield full_response
                except TypeError:
                    # full_response might not be iterable
                    if full_response is not None:
                        yield full_response
                
            except Exception as exc:
                logger.exception(f"Ollama async _stream instrumentation error: {exc}")
                _set_exception_attributes(span, exc)
                raise
    
    return wrapped


def _extract_endpoint(url: str) -> str:
    """Extract the API endpoint path from a full URL."""
    if not url:
        return ""
    # Extract path after domain
    parts = url.split("/api/", 1)
    if len(parts) > 1:
        return f"/api/{parts[1]}"
    return url


def _set_request_model_attribute(span, kwargs: dict) -> None:
    """Try to extract and set the model from request kwargs."""
    # Look for model in different places
    if "json" in kwargs and isinstance(kwargs["json"], dict):
        model = kwargs["json"].get("model")
        if model:
            span.set_attribute("gen_ai.request.model", str(model))


def _consume_response_stream(response_iter) -> list:
    """Fully consume a sync response stream/iterator to get complete response."""
    if response_iter is None:
        return []
    
    try:
        if hasattr(response_iter, '__iter__'):
            # It's an iterator, consume it fully
            items = []
            for item in response_iter:
                items.append(item)
            return items
        else:
            # It's a single response object
            return [response_iter]
    except Exception as e:
        logger.warning(f"Error consuming response stream: {e}")
        return []


async def _consume_async_response_stream(response_iter) -> list:
    """Fully consume an async response stream/iterator to get complete response."""
    if response_iter is None:
        return []
    
    try:
        if hasattr(response_iter, '__aiter__'):
            # It's an async iterator, consume it fully
            items = []
            async for item in response_iter:
                items.append(item)
            return items
        else:
            # It's a single response object
            return [response_iter]
    except Exception as e:
        logger.warning(f"Error consuming async response stream: {e}")
        return []


def _set_response_attributes_from_stream(span, response_items: list) -> None:
    """Extract and set response attributes from a stream of response chunks.
    
    In Ollama, the timing metadata is typically in the LAST chunk of the streamed
    response. This function processes all chunks to find and extract timing data.
    """
    if not response_items:
        return
    
    # Process all items to find timing metadata
    # Typically the last item has the full response_metadata
    for item in response_items:
        if item is None:
            continue
        
        # Check if this item contains timing data
        model = _extract_value(item, "model")
        if model:
            span.set_attribute("gen_ai.response.model", str(model))
        
        # Extract timing fields
        timing_fields = {
            "total_duration": "ollama.total_duration",
            "load_duration": "ollama.load_duration",
            "prompt_eval_duration": "ollama.prompt_eval_duration",
            "eval_duration": "ollama.eval_duration",
            "eval_count": "ollama.eval_count",
        }
        
        for attr_name, span_attr in timing_fields.items():
            value = _extract_value(item, attr_name)
            if value is not None:
                try:
                    int_val = int(value) if isinstance(value, (str, float, int)) else value
                    span.set_attribute(span_attr, int_val)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {attr_name}={value} to int: {e}")


def _set_response_attributes(span, result: Any) -> None:
    """Extract and set response attributes from the Ollama response."""
    if result is None:
        return
    
    # Extract model from response
    model = _extract_value(result, "model")
    if model:
        span.set_attribute("gen_ai.response.model", str(model))

    # Ollama returns these durations in nanoseconds from response_metadata.
    # These fields are CRITICAL for the observability goal.
    timing_fields = {
        "total_duration": "total_duration",
        "load_duration": "load_duration", 
        "prompt_eval_duration": "prompt_eval_duration",
        "eval_duration": "eval_duration",
        "eval_count": "eval_count",
    }
    
    for attr_name, span_attr in timing_fields.items():
        value = _extract_value(result, attr_name)
        if value is not None:
            try:
                # Convert to int for duration fields (in nanoseconds)
                int_val = int(value) if isinstance(value, (str, float, int)) else value
                span.set_attribute(span_attr, int_val)
                logger.debug(f"Set span attribute {span_attr}={int_val} from response")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert {attr_name}={value} to int: {e}")
        else:
            logger.debug(f"Response attribute {attr_name} not found in response")


def _set_exception_attributes(span, exc: Exception) -> None:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            span.set_attribute("http.status_code", int(status_code))
        except Exception:
            span.set_attribute("http.status_code", str(status_code))

    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def _extract_value(result: Any, key: str) -> Any:
    """Robustly extract a value from an Ollama response object.
    
    Ollama responses can be dicts, Pydantic models, or custom objects.
    Timing fields are nested under response_metadata.
    """
    if result is None:
        return None

    # Try direct dict access (result might be a dict-like object)
    if isinstance(result, dict):
        # First check top-level (for direct fields like 'model')
        if key in result:
            return result[key]
        
        # Then check response_metadata nested dict
        metadata = result.get("response_metadata")
        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]
    
    # Try attribute access  
    if hasattr(result, key):
        value = getattr(result, key)
        if value is not None:
            return value
    
    # Try model_dump for Pydantic v2
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            if isinstance(dumped, dict):
                if key in dumped:
                    return dumped[key]
                metadata = dumped.get("response_metadata")
                if isinstance(metadata, dict) and key in metadata:
                    return metadata[key]
        except Exception:
            pass
    
    # Try dict() for Pydantic v1
    model_dict = getattr(result, "dict", None)
    if callable(model_dict):
        try:
            dumped = model_dict()
            if isinstance(dumped, dict):
                if key in dumped:
                    return dumped[key]
                metadata = dumped.get("response_metadata")
                if isinstance(metadata, dict) and key in metadata:
                    return metadata[key]
        except Exception:
            pass
    
    # Try response_metadata attribute directly
    metadata_attr = getattr(result, "response_metadata", None)
    if isinstance(metadata_attr, dict):
        if key in metadata_attr:
            return metadata_attr[key]
    
    # Check if response_metadata object has dotted access
    if metadata_attr is not None and hasattr(metadata_attr, key):
        return getattr(metadata_attr, key)
    
    return None


def _build_request_url(client: Any, endpoint_path: str) -> str:
    base_url = None

    for attr_name in ("host", "_host", "base_url", "_base_url"):
        value = getattr(client, attr_name, None)
        if value:
            base_url = str(value)
            break

    if base_url is None:
        http_client = getattr(client, "_client", None)
        if http_client is not None:
            base_candidate = getattr(http_client, "base_url", None)
            if base_candidate:
                base_url = str(base_candidate)

    if not base_url:
        return ""

    if not base_url.endswith("/"):
        base_url = f"{base_url}/"

    return urljoin(base_url, endpoint_path.lstrip("/"))
