"""Tests for the model-written-SQL verification route.

Two things are worth asserting here and neither needs a model. The comparison must
judge each answer type on its own terms, and the read-only gate must refuse
everything a model could emit that would write to the database, because the query it
runs is untrusted text.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.helper_ollama import _extract_sql
from src.utils.helper_postgres import assert_readonly_select
from src.commands.sql_verification import compare


class CompareTest(unittest.TestCase):
    def test_matching_count_agrees(self):
        agrees, _detail = compare("count", "490", 490)
        self.assertTrue(agrees)

    def test_mismatched_count_disagrees(self):
        agrees, _detail = compare("count", "999999", 490)
        self.assertFalse(agrees)

    def test_presence_boolean_true_maps_to_yes(self):
        self.assertTrue(compare("presence", "Yes (3 matching lines)", True)[0])
        self.assertFalse(compare("presence", "No (0 matching lines)", True)[0])

    def test_presence_count_maps_through_its_sign(self):
        self.assertTrue(compare("presence", "Yes (3 matching lines)", 3)[0])
        self.assertTrue(compare("presence", "No (0 matching lines)", 0)[0])
        self.assertFalse(compare("presence", "Yes (3 matching lines)", 0)[0])

    def test_lookup_compares_text_ignoring_surrounding_space(self):
        self.assertTrue(compare("line_lookup", "a log line", "  a log line  ")[0])
        self.assertFalse(compare("line_lookup", "a log line", "another line")[0])

    def test_no_rows_never_agrees(self):
        for answer_type, gold in (("count", "0"), ("presence", "No")):
            self.assertFalse(compare(answer_type, gold, None)[0])

    def test_semantic_answers_cannot_be_settled_by_a_query(self):
        agrees, detail = compare("explanation", "some prose", "some prose")
        self.assertFalse(agrees)
        self.assertIn("cannot be settled", detail)

    def test_non_numeric_result_for_a_count_disagrees(self):
        agrees, detail = compare("count", "3", "three")
        self.assertFalse(agrees)
        self.assertIn("not a count", detail)


class ReadOnlyGateTest(unittest.TestCase):
    def test_plain_select_is_allowed(self):
        assert_readonly_select("SELECT COUNT(*) FROM lines WHERE dataset = 'linux'")

    def test_cte_is_allowed(self):
        assert_readonly_select("WITH m AS (SELECT 1) SELECT * FROM m")

    def test_trailing_semicolon_is_tolerated(self):
        assert_readonly_select("SELECT 1;")

    def test_stacked_statements_are_refused(self):
        with self.assertRaises(ValueError):
            assert_readonly_select("SELECT 1; DROP TABLE lines")

    def test_writes_are_refused(self):
        for statement in (
            "DELETE FROM lines",
            "UPDATE lines SET text = 'x'",
            "INSERT INTO lines VALUES (1)",
            "DROP TABLE lines",
            "TRUNCATE lines",
            "ALTER TABLE lines ADD COLUMN x INT",
            "CREATE TABLE t (a INT)",
            "GRANT ALL ON lines TO public",
            "COPY lines TO '/tmp/x'",
        ):
            with self.assertRaises(ValueError, msg=statement):
                assert_readonly_select(statement)

    def test_filesystem_and_sleep_functions_are_refused(self):
        with self.assertRaises(ValueError):
            assert_readonly_select("SELECT pg_read_file('/etc/passwd')")
        with self.assertRaises(ValueError):
            assert_readonly_select("SELECT pg_sleep(100)")

    def test_empty_query_is_refused(self):
        with self.assertRaises(ValueError):
            assert_readonly_select("   ")


class SqlExtractionTest(unittest.TestCase):
    def test_bare_statement_passes_through(self):
        self.assertEqual(_extract_sql("SELECT 1"), "SELECT 1")

    def test_fenced_sql_is_unwrapped(self):
        self.assertEqual(_extract_sql("```sql\nSELECT 1\n```"), "SELECT 1")

    def test_unlabelled_fence_is_unwrapped(self):
        self.assertEqual(_extract_sql("```\nSELECT 1\n```"), "SELECT 1")

    def test_trailing_semicolon_is_stripped(self):
        self.assertEqual(_extract_sql("SELECT 1;"), "SELECT 1")


if __name__ == "__main__":
    unittest.main()
