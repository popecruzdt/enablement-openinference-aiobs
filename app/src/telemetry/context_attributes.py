"""Copy the OpenInference context attributes (session id, user id, tags, metadata) onto every span.

OpenInference's own spans already get these attributes from `using_session(...)` and friends.
The app's manual spans (workflow, Weaviate, Ollama client) are created with the plain
OpenTelemetry API and would not, so this processor adds them when a span starts.
"""

from __future__ import annotations

import logging

from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger("ai-travel-advisor")


class ContextAttributesSpanProcessor(SpanProcessor):
    def on_start(self, span, parent_context=None):
        try:
            from openinference.instrumentation import get_attributes_from_context

            for key, value in get_attributes_from_context():
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
