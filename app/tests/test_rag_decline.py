"""RAG answers a fixed sentence when the destination has no page. Run from app/ with the app requirements (see test_retrieval.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import NO_INFORMATION_ANSWER  # noqa: E402
from src.rag import RAGPipeline, RetrievalResult  # noqa: E402


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return type("R", (), {"content": "model answer"})()


def pipeline(result):
    rag = object.__new__(RAGPipeline)
    rag.rag_chain = True
    rag.llm = FakeLLM()
    rag.retrieve_context = lambda destination: result
    rag.load_rag_prompt_template = lambda: "{context} {destination}"
    rag.settings = type("S", (), {"ai_model": "m"})()
    return rag


class RagDeclineTest(unittest.TestCase):
    def test_unknown_destination_gets_the_fixed_sentence_without_calling_the_model(self):
        rag = pipeline(RetrievalResult(contexts=["Destination Singapore text"], destination_found=False))
        self.assertEqual(rag.get_travel_advice("SF"), NO_INFORMATION_ANSWER)
        self.assertEqual(rag.llm.calls, 0)

    def test_known_destination_is_answered_by_the_model(self):
        rag = pipeline(RetrievalResult(contexts=["Boston text"], destination_found=True))
        self.assertEqual(rag.get_travel_advice("Boston"), "model answer")
        self.assertEqual(rag.llm.calls, 1)

    def test_drift_still_reaches_the_model(self):
        rag = pipeline(RetrievalResult(contexts=[], drift_active=True, destination_found=True))
        self.assertEqual(rag.get_travel_advice("Boston"), "model answer")


if __name__ == "__main__":
    unittest.main()
