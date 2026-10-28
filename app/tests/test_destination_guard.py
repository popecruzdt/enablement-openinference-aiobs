"""Tests for the agent's destination guard. Run from app/ with the app requirements (see test_retrieval.py)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import destination_from_request  # noqa: E402


class DestinationFromRequestTest(unittest.TestCase):
    def test_invented_destination_is_replaced_by_the_request(self):
        self.assertEqual(destination_from_request("Tokyo", "SF"), "SF")

    def test_destination_named_in_the_request_is_kept(self):
        self.assertEqual(destination_from_request("Boston", "Boston"), "Boston")
        self.assertEqual(destination_from_request("new york", "New York"), "new york")
        self.assertEqual(destination_from_request("San Francisco", "San Francisco, CA"), "San Francisco")

    def test_partial_word_is_not_a_match(self):
        self.assertEqual(destination_from_request("Rome", "Romeo"), "Romeo")

    def test_empty_model_destination_uses_the_request(self):
        self.assertEqual(destination_from_request("", "Lisbon"), "Lisbon")


if __name__ == "__main__":
    unittest.main()
