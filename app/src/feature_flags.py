"""
Feature flag management using the OpenFeature SDK with an in-process InMemoryProvider.

The InMemoryProvider requires no sidecar or external service. To swap to a remote
provider (flagd, LaunchDarkly, Dynatrace Feature Flags), replace set_provider() call
and remove this module's direct _provider reference — get_embedding_override() and
set_embedding_override() remain unchanged.
"""

import logging

from openfeature import api as openfeature_api
from openfeature.event import ProviderEventDetails
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider

logger = logging.getLogger(__name__)

EMBEDDING_OVERRIDE_FLAG = "embedding-model-override"

_provider = InMemoryProvider({
    EMBEDDING_OVERRIDE_FLAG: InMemoryFlag(
        default_variant="default",
        variants={"default": ""},  # empty string = use the configured AI_EMBEDDING_MODEL
    )
})
openfeature_api.set_provider(_provider)


def set_embedding_override(model: str) -> None:
    """
    Set or clear the embedding model override flag.

    Args:
        model: Ollama model name to override with (e.g. 'gemma2:2b'), or empty string to reset.
    """
    if model:
        logger.info(f"Feature flag '{EMBEDDING_OVERRIDE_FLAG}' set to override: {model}")
        _provider._flags[EMBEDDING_OVERRIDE_FLAG] = InMemoryFlag(
            default_variant="override",
            variants={"default": "", "override": model},
        )
    else:
        logger.info(f"Feature flag '{EMBEDDING_OVERRIDE_FLAG}' reset to default")
        _provider._flags[EMBEDDING_OVERRIDE_FLAG] = InMemoryFlag(
            default_variant="default",
            variants={"default": ""},
        )
    _provider.emit_provider_configuration_changed(
        ProviderEventDetails(flags_changed=[EMBEDDING_OVERRIDE_FLAG])
    )


def get_embedding_override() -> str:
    """
    Evaluate the embedding model override flag.

    Returns:
        The override model name (e.g. 'gemma2:2b'), or '' if the flag is in its default state.
    """
    return openfeature_api.get_client().get_string_value(EMBEDDING_OVERRIDE_FLAG, "")
