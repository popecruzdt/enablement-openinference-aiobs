"""Emit an OpenTelemetry log record whenever the embedding-model-override flag changes.

OpenFeature announces flag changes with the provider event PROVIDER_CONFIGURATION_CHANGED. This
module listens for it and, when the flag value actually changed, emits one OTel log record
(event_name `feature_flag.change`) over OTLP. Dynatrace OpenPipeline extracts a Davis event from
that record; the app itself stays vendor-neutral. Attribute names follow the OpenTelemetry
`feature_flag.*` semantic conventions where one exists (`feature_flag.key`,
`feature_flag.provider.name`, `feature_flag.result.value`); `feature_flag.previous_value` and
`feature_flag.result.enabled` are app-specific, and `feature_flag.result.variant` holds the embedding model
actually in use (the override, or the configured model when the flag is at its default).
When OneAgent is detected, its entity IDs are added to the log resource for entity linking in OpenPipeline.

The Logs SDK is only available under `opentelemetry.sdk._logs` in the pinned SDK (1.41), so the
setup is guarded: any failure disables the events and never affects request handling.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

from opentelemetry import trace

from src.config import get_settings
from src.feature_flags import EMBEDDING_OVERRIDE_FLAG, get_embedding_override

logger = logging.getLogger("ai-travel-advisor")

EVENT_NAME = "feature_flag.change"

# When OneAgent is injected into the process, this (virtual) file holds the path of a JSON file
# with the process's Dynatrace entity IDs (process group instance, host, host group, ...).
_ONEAGENT_METADATA_FILE = "dt_metadata_e617c525669e072eebe3d0f08212e8f2.json"

_lock = threading.Lock()
_last_value: Optional[str] = None
_otel_logger = None
_registered = False


def _read_oneagent_metadata() -> dict:
    """Dynatrace entity IDs for this process from OneAgent metadata enrichment, or {} if not detected."""
    try:
        with open(_ONEAGENT_METADATA_FILE) as f:
            metadata_path = f.read().strip()
        with open(metadata_path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool))}
    except Exception:
        return {}


def _on_configuration_changed(event_details) -> None:
    """OpenFeature PROVIDER_CONFIGURATION_CHANGED handler: emit a log record if the value changed."""
    global _last_value
    try:
        flags = getattr(event_details, "flags_changed", None) or []
        if flags and EMBEDDING_OVERRIDE_FLAG not in flags:
            return
        new_value = get_embedding_override()
        with _lock:
            previous = _last_value
            if new_value == previous:
                return
            _last_value = new_value
        _emit(previous or "", new_value, getattr(event_details, "provider_name", None))
    except Exception:
        logger.debug("Feature flag change event failed", exc_info=True)


def _emit(previous: str, new_value: str, provider_name: Optional[str]) -> None:
    if _otel_logger is None:
        return
    from opentelemetry._logs import LogRecord, SeverityNumber

    attributes = {
        "feature_flag.key": EMBEDDING_OVERRIDE_FLAG,
        "feature_flag.result.value": new_value,
        "feature_flag.previous_value": previous,
        # Model actually in use: the override, or the configured model when the flag is at its default.
        "feature_flag.result.variant": new_value or get_settings().ai_embedding_model,
        # True while an override is active (flag value is not the default '').
        "feature_flag.result.enabled": bool(new_value),
    }
    if provider_name:
        attributes["feature_flag.provider.name"] = provider_name
    now = time.time_ns()
    _otel_logger.emit(
        LogRecord(
            timestamp=now,
            observed_timestamp=now,
            event_name=EVENT_NAME,
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            body=f"Feature flag '{EMBEDDING_OVERRIDE_FLAG}' changed: '{previous}' -> '{new_value}'",
            attributes=attributes,
        )
    )
    logger.info(f"Emitted feature flag change event for '{EMBEDDING_OVERRIDE_FLAG}'")


def setup_feature_flag_events(otlp_endpoint: str, insecure: bool, log_exporter=None) -> None:
    """Create the OTel LoggerProvider and register the OpenFeature change handler (idempotent).

    `log_exporter` lets tests inject an in-memory exporter instead of the OTLP gRPC exporter.
    """
    global _otel_logger, _last_value, _registered
    if _registered:
        return
    try:
        from openfeature import api as openfeature_api
        from openfeature.event import ProviderEvent
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        if log_exporter is None:
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

            log_exporter = OTLPLogExporter(endpoint=otlp_endpoint, insecure=insecure)

        resource = getattr(trace.get_tracer_provider(), "resource", None) or Resource.create(
            {"service.name": "ai-travel-advisor"}
        )
        # OneAgent entity IDs let OpenPipeline link the extracted event to the process and host.
        # service.name stays on the resource either way, so the event is always attributable to the service.
        oneagent_metadata = _read_oneagent_metadata()
        if oneagent_metadata:
            resource = resource.merge(Resource.create(oneagent_metadata))
        provider = LoggerProvider(resource=resource)
        provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        _otel_logger = provider.get_logger("ai-travel-advisor.feature-flags")
        _last_value = get_embedding_override()
        openfeature_api.add_handler(ProviderEvent.PROVIDER_CONFIGURATION_CHANGED, _on_configuration_changed)
        _registered = True
        logger.info(
            "Enabled feature flag change events (OTel log records); "
            f"OneAgent metadata {'detected' if oneagent_metadata else 'not detected'}"
        )
    except Exception as e:
        logger.warning(f"Could not enable feature flag change events: {e}")
