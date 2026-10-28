"""OpenInference retrieval attributes for the span that returns the knowledge-base context.

Arize AX's retrieval view and RAG evaluators (retrieval relevance, QA correctness) read
`input.value` and `retrieval.documents.<i>.document.*` from a RETRIEVER span. `RetrievedDocument`
is the app's neutral record; `record_retrieval_documents` writes it onto the span. It never raises:
tracing must not affect a request.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger("ai-travel-advisor")

# Per-document cap, to keep span size bounded (a KB chunk is about 500 characters).
MAX_DOCUMENT_CHARS = 2000

_SPAN_KIND = "openinference.span.kind"


@dataclass
class RetrievedDocument:
    content: str
    id: Optional[str] = None
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def record_retrieval_documents(span, query: str, documents: Iterable[RetrievedDocument]) -> None:
    """Mark `span` as a RETRIEVER and attach the query and the documents that were used."""
    try:
        if span is None or not span.is_recording():
            return
        span.set_attribute(_SPAN_KIND, "RETRIEVER")
        span.set_attribute("input.value", query)
        span.set_attribute("input.mime_type", "text/plain")
        for index, doc in enumerate(documents):
            prefix = f"retrieval.documents.{index}.document"
            span.set_attribute(f"{prefix}.content", (doc.content or "")[:MAX_DOCUMENT_CHARS])
            if doc.id:
                span.set_attribute(f"{prefix}.id", doc.id)
            if doc.score is not None:
                span.set_attribute(f"{prefix}.score", float(doc.score))
            if doc.metadata:
                span.set_attribute(f"{prefix}.metadata", json.dumps(doc.metadata, ensure_ascii=False))
    except Exception:
        logger.debug("Could not record retrieval documents", exc_info=True)
