"""Tests for the pure parsing logic in ``src.utils.helper_vllm``.

No network and no model server: these exercise ``_parse_dimension_verdicts``
against hand-written completions, the same way a real ``groundedness_model``
response would be shaped once ``_strip_thinking`` has already run over it.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.helper_vllm import _parse_dimension_verdicts


DIMENSIONS = [
    ("grounded", "Is the answer grounded?"),
    ("correct", "Is the answer correct?"),
    ("relevant", "Is the answer relevant?"),
]


class ParseDimensionVerdictsTest(unittest.TestCase):
    def test_parses_every_dimension_in_order(self):
        text = (
            "1) yes: fully grounded in the evidence.\n"
            "2) no: misreads the entity.\n"
            "3) partial: only half addresses the question.\n"
        )
        checks = _parse_dimension_verdicts(text, DIMENSIONS)
        self.assertEqual(
            checks,
            [
                {
                    "dimension": "grounded",
                    "verdict": "yes",
                    "detail": "fully grounded in the evidence.",
                },
                {
                    "dimension": "correct",
                    "verdict": "no",
                    "detail": "misreads the entity.",
                },
                {
                    "dimension": "relevant",
                    "verdict": "partial",
                    "detail": "only half addresses the question.",
                },
            ],
        )

    def test_matches_by_leading_number_not_line_order(self):
        text = "2) no: wrong.\n1) yes: right.\n3) yes: fine.\n"
        checks = _parse_dimension_verdicts(text, DIMENSIONS)
        self.assertEqual(checks[0]["verdict"], "yes")
        self.assertEqual(checks[1]["verdict"], "no")
        self.assertEqual(checks[2]["verdict"], "yes")

    def test_missing_dimension_falls_back_to_partial(self):
        text = "1) yes: fine.\n3) yes: fine too.\n"
        checks = _parse_dimension_verdicts(text, DIMENSIONS)
        self.assertEqual(checks[1], {"dimension": "correct", "verdict": "partial"})

    def test_unparseable_completion_yields_all_partial(self):
        text = "I am not sure how to answer this."
        checks = _parse_dimension_verdicts(text, DIMENSIONS)
        self.assertTrue(all(check["verdict"] == "partial" for check in checks))
        self.assertTrue(all("detail" not in check for check in checks))

    def test_first_occurrence_of_a_repeated_number_wins(self):
        text = "1) yes: first answer.\n1) no: a stray repeat.\n2) yes: fine.\n3) yes: fine.\n"
        checks = _parse_dimension_verdicts(text, DIMENSIONS)
        self.assertEqual(checks[0]["verdict"], "yes")
        self.assertEqual(checks[0]["detail"], "first answer.")


if __name__ == "__main__":
    unittest.main()
