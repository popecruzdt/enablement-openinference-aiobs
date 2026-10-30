"""Tests for the sign-in user list. Run from app/: python3 -m unittest discover -s tests"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.users import ANONYMOUS_USER_ID, UserDirectory, resolve_request_user  # noqa: E402


class UserDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.file = os.path.join(self.dir.name, "users.txt")

    def tearDown(self):
        self.dir.cleanup()

    def write(self, text):
        with open(self.file, "w", encoding="utf-8") as handle:
            handle.write(text)
        # a rewrite within the same clock tick would keep the same mtime; the size changes in these tests
        os.utime(self.file, None)

    def test_listed_ids_resolve_case_insensitively_with_file_casing(self):
        self.write("# comment\nDoug\n\ngabe  # trailing comment\n")
        users = UserDirectory(self.dir.name)
        self.assertEqual(users.resolve("doug"), "Doug")
        self.assertEqual(users.resolve("  GABE "), "gabe")

    def test_unknown_and_malformed_ids_do_not_resolve(self):
        self.write("doug\n")
        users = UserDirectory(self.dir.name)
        for bad in ("butthead", "", None, "doug smith", "doug;drop", "x" * 65, "../etc/passwd"):
            self.assertIsNone(users.resolve(bad), bad)

    def test_anonymous_and_invalid_lines_in_the_file_are_ignored(self):
        self.write("anonymous\nAnonymous\nbad id\ngood\n")
        users = UserDirectory(self.dir.name)
        self.assertIsNone(users.resolve("anonymous"))
        self.assertIsNone(users.resolve("bad id"))
        self.assertEqual(users.resolve("good"), "good")

    def test_missing_file_means_nobody_can_sign_in(self):
        users = UserDirectory(self.dir.name)
        self.assertIsNone(users.resolve("doug"))

    def test_edits_are_picked_up_without_restart(self):
        self.write("doug\n")
        users = UserDirectory(self.dir.name)
        self.assertIsNone(users.resolve("tpc"))
        self.write("doug\ntpc\n")
        self.assertEqual(users.resolve("tpc"), "tpc")
        self.write("tpc\n")
        self.assertIsNone(users.resolve("doug"))

    def test_request_user_falls_back_to_anonymous(self):
        self.write("doug\n")
        users = UserDirectory(self.dir.name)
        self.assertEqual(resolve_request_user(users, ""), ANONYMOUS_USER_ID)
        self.assertEqual(resolve_request_user(users, None), ANONYMOUS_USER_ID)
        self.assertEqual(resolve_request_user(users, "butthead"), ANONYMOUS_USER_ID)
        self.assertEqual(resolve_request_user(users, "DOUG"), "doug")


if __name__ == "__main__":
    unittest.main()
