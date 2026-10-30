"""Tests for the retrieval span attributes. Run from app/: uv run --with opentelemetry-sdk --with opentelemetry-exporter-otlp-proto-grpc python3 -m unittest discover -s tests"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import NonRecordingSpan, INVALID_SPAN_CONTEXT  # noqa: E402

from src.telemetry.retrieval import MAX_DOCUMENT_CHARS, RetrievedDocument, record_retrieval_documents  # noqa: E402


class RecordRetrievalDocumentsTest(unittest.TestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = provider.get_tracer("test")

    def record(self, query, docs):
        with self.tracer.start_as_current_span("weaviate.retrieve") as span:
            record_retrieval_documents(span, query, docs)
        return dict(self.exporter.get_finished_spans()[0].attributes)

    def test_kind_query_and_indexed_documents(self):
        attrs = self.record("Boston", [
            RetrievedDocument("first", id="u1", score=2.5, metadata={"title": "boston", "source": "/a/boston.html"}),
            RetrievedDocument("second", id="u2", score=1.0),
        ])
        self.assertEqual(attrs["openinference.span.kind"], "RETRIEVER")
        self.assertEqual(attrs["input.value"], "Boston")
        self.assertEqual(attrs["input.mime_type"], "text/plain")
        self.assertEqual(attrs["retrieval.documents.0.document.content"], "first")
        self.assertEqual(attrs["retrieval.documents.0.document.id"], "u1")
        self.assertEqual(attrs["retrieval.documents.0.document.score"], 2.5)
        self.assertEqual(json.loads(attrs["retrieval.documents.0.document.metadata"]),
                         {"title": "boston", "source": "/a/boston.html"})
        self.assertEqual(attrs["retrieval.documents.1.document.content"], "second")
        self.assertNotIn("retrieval.documents.1.document.metadata", attrs)

    def test_no_documents_still_a_retriever_with_the_query(self):
        attrs = self.record("San Francisco", [])
        self.assertEqual(attrs["openinference.span.kind"], "RETRIEVER")
        self.assertEqual(attrs["input.value"], "San Francisco")
        self.assertFalse([k for k in attrs if k.startswith("retrieval.documents")])

    def test_content_is_truncated(self):
        attrs = self.record("x", [RetrievedDocument("a" * (MAX_DOCUMENT_CHARS + 500))])
        self.assertEqual(len(attrs["retrieval.documents.0.document.content"]), MAX_DOCUMENT_CHARS)

    def test_non_recording_span_is_a_no_op(self):
        record_retrieval_documents(NonRecordingSpan(INVALID_SPAN_CONTEXT), "x", [RetrievedDocument("a")])
        record_retrieval_documents(None, "x", [RetrievedDocument("a")])

    def test_bad_document_does_not_raise(self):
        with self.tracer.start_as_current_span("s") as span:
            record_retrieval_documents(span, "x", [object()])


if __name__ == "__main__":
    unittest.main()
