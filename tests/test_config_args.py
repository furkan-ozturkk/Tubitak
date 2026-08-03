"""Tests for the CLI surface's own validations.

Each of ``config.args``'s three validations exists to turn a silent wrong answer into
an immediate failure, so each is asserted to actually fail rather than to merely be
present.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.args import DEFAULT_DATASET, _default_full_dataset, args_parser


class PathResolutionTest(unittest.TestCase):
    def test_questions_defaults_to_the_dataset(self):
        args = args_parser(["--dataset", "/tmp/questions.json"])
        self.assertEqual(args.questions, ["/tmp/questions.json"])

    def test_review_out_defaults_to_overwriting_the_dataset(self):
        args = args_parser(["--dataset", "/tmp/questions.json"])
        self.assertEqual(args.review_out, Path("/tmp/questions.json"))

    def test_explicit_questions_is_not_overridden(self):
        args = args_parser(["--questions", "a.json", "b.json"])
        self.assertEqual(args.questions, ["a.json", "b.json"])

    def test_full_moves_off_the_official_output(self):
        args = args_parser(["--command", "generate", "--full"])
        self.assertEqual(args.dataset, _default_full_dataset())
        self.assertNotEqual(args.dataset, DEFAULT_DATASET)

    def test_full_honours_an_explicit_dataset(self):
        args = args_parser(
            ["--command", "generate", "--full", "--dataset", "/tmp/x.json"]
        )
        self.assertEqual(args.dataset, Path("/tmp/x.json"))

    def test_default_pass_writes_the_official_output(self):
        args = args_parser(["--command", "generate"])
        self.assertEqual(args.dataset, DEFAULT_DATASET)


class ModelSeparationTest(unittest.TestCase):
    def test_identical_models_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            args_parser(["--gold_draft_model", "x:1b", "--groundedness_model", "x:1b"])
        self.assertIn("different families", str(caught.exception))

    def test_different_models_are_accepted(self):
        args = args_parser(
            ["--gold_draft_model", "a:1b", "--groundedness_model", "b:2b"]
        )
        self.assertEqual(args.gold_draft_model, "a:1b")


class TierKnobTest(unittest.TestCase):
    def test_zero_window_size_is_rejected(self):
        with self.assertRaises(ValueError):
            args_parser(["--window_size", "0"])

    def test_zero_questions_per_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            args_parser(["--questions_per_dataset", "0"])

    def test_zero_parallel_calls_is_rejected(self):
        with self.assertRaises(ValueError):
            args_parser(["--max_parallel_model_calls", "0"])

    def test_negative_min_matches_is_rejected(self):
        with self.assertRaises(ValueError):
            args_parser(["--min_matches", "-1"])

    def test_zero_min_matches_is_allowed(self):
        args = args_parser(["--min_matches", "0"])
        self.assertEqual(args.min_matches, 0)

    def test_test_fraction_of_one_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            args_parser(["--test_fraction", "1.0"])
        self.assertIn("no dev set", str(caught.exception))

    def test_negative_test_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            args_parser(["--test_fraction", "-0.1"])

    def test_omitted_knobs_stay_none_so_dataclasses_win(self):
        args = args_parser([])
        for name in (
            "min_matches",
            "max_cited_lines",
            "window_size",
            "questions_per_dataset",
            "min_sentences",
            "test_fraction",
            "max_parallel_model_calls",
            "max_retries",
            "backoff_base_seconds",
            "temperature",
            "target_total_questions",
            "reviewer",
            "created_at",
        ):
            self.assertIsNone(getattr(args, name), f"{name} should default to None")


if __name__ == "__main__":
    unittest.main()
