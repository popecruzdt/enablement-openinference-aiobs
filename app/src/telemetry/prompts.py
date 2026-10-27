"""OpenInference prompt-template and request-context helpers.

`prompt_template_context(...)` records the raw template, a version and the variable values on the
LLM span started inside the block (`llm.prompt_template.*`), which Arize AX shows next to the rendered
prompt. `request_context(...)` records the session id and the user id on every span of a request.

Both fall back to a no-op if the OpenInference helpers are not installed.
"""

from __future__ import annotations

import hashlib
import uuid
from contextlib import nullcontext
from typing import Any, Dict, Optional

# The user id of a request that is not signed in (sign-in is optional, see src/users.py).
ANONYMOUS_USER_ID = "anonymous"

# A variable value (for example the retrieved RAG context) is capped so that a larger CHUNK_SIZE or
# RETRIEVAL_K cannot make the span attribute grow without bound. Today's contexts are well below it.
MAX_VARIABLE_CHARS = 8000


def prompt_version(template: str) -> str:
    """Short, stable version for a template: it changes whenever the prompt text is edited."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]


def _cap(value: Any) -> Any:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= MAX_VARIABLE_CHARS:
        return text
    return f"{text[:MAX_VARIABLE_CHARS]}… [truncated, {len(text)} characters]"


def prompt_template_context(template: str, variables: Optional[Dict[str, Any]] = None):
    """Context manager: LLM calls made inside it carry this prompt template, its version and variables."""
    try:
        from openinference.instrumentation import using_prompt_template

        return using_prompt_template(
            template=template,
            version=prompt_version(template),
            variables={name: _cap(value) for name, value in (variables or {}).items()},
        )
    except ImportError:
        return nullcontext()


def request_context(session_id: str = "", user_id: str = ANONYMOUS_USER_ID):
    """Context manager for one request: session id (generated when empty) and the user id."""
    try:
        from openinference.instrumentation import using_attributes

        return using_attributes(
            session_id=session_id.strip()[:128] or uuid.uuid4().hex,
            user_id=user_id or ANONYMOUS_USER_ID,
        )
    except ImportError:
        return nullcontext()
