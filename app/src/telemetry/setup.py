"""Tracing setup: an OpenTelemetry SDK tracer provider that exports over OTLP/gRPC.

The app creates the provider and exporter itself. OpenInference, the gRPC client
instrumentation and the manual spans all send their spans through this provider.
"""

from __future__ import annotations

import logging
from typing import Iterable

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger("ai-travel-advisor")


def setup_tracing(
    otlp_endpoint: str,
    insecure: bool,
    service_name: str,
    project_name: str,
    span_processors: Iterable[SpanProcessor] = (),
) -> TracerProvider:
    """Create the global tracer provider.

    `openinference.project.name` groups the traces into a project in Arize AX.
    The batch exporter is registered first, then `span_processors` in the order given.
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "openinference.project.name": project_name,
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=insecure))
    )
    for processor in span_processors:
        provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logger.info(f"OpenTelemetry tracing enabled, exporting to {otlp_endpoint}")
    return provider
