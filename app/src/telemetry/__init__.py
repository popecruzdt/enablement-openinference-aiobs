"""Telemetry helpers for custom OpenTelemetry instrumentation."""

from .ollama_middle_span import instrument_ollama_middle_span

__all__ = ["instrument_ollama_middle_span"]
