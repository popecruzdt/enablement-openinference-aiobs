"""Copy the session id and user id from the OpenInference context onto every span.

OpenInference's own spans already get them from `using_attributes(...)`. The app's manual spans
(workflow, Weaviate, Ollama client) are created with the plain OpenTelemetry API and would not, so
this processor adds them when a span starts. Only these two attributes are copied: prompt template
attributes belong on the LLM span alone, not on the manual spans created inside the call.
"""

from __future__ import annotations

import logging

from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger("ai-travel-advisor")

_COPIED_ATTRIBUTES = ("session.id", "user.id")


class ContextAttributesSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context=None):
        try:
            from openinference.instrumentation import get_attributes_from_context

            for key, value in get_attributes_from_context():
                if key in _COPIED_ATTRIBUTES:
                    span.set_attribute(key, value)
        except Exception:
            # Tracing must never affect request handling.
            pass

    def on_end(self, span):
        return

    def shutdown(self):
        return

    def force_flush(self, timeout_millis=30000):
        return True
