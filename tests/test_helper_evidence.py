"""Tests for the record-building primitives shared by the three tiers."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.corpus_loader import lines_from_bytes, sha256_line
from src.utils.helper_evidence import (
    evidence_ref,
    gold_provenance,
    slugify,
    split_sentences,
)


class SlugifyTest(unittest.TestCase):
    def test_produces_only_schema_legal_characters(self):
        import re

        for literal in (
            "authentication failure",
            "[notice]",
            "status: 200",
            "POSSIBLE BREAK-IN ATTEMPT",
            "double-hummer alignment exceptions",
        ):
            self.assertRegex(slugify(literal), r"^[a-z0-9_]+$", literal)

    def test_collapses_runs_and_trims_edges(self):
        self.assertEqual(slugify("  a -- b  "), "a_b")

    def test_is_case_insensitive(self):
        self.assertEqual(slugify("ERROR"), slugify("error"))


class EvidenceRefTest(unittest.TestCase):
    def test_hash_matches_the_cited_text(self):
        ref = evidence_ref("linux", 7, "a log line", "linux:count:x")
        self.assertEqual(ref["line_hash"], sha256_line("a log line"))

    def test_id_embeds_dataset_and_zero_padded_line(self):
        ref = evidence_ref("hdfs", 42, "text", "hdfs:count:x")
        self.assertTrue(ref["id"].startswith("hdfs:line:00000042:"))

    def test_id_prefix_round_trips_to_the_dataset_key(self):
        ref = evidence_ref("openssh", 1, "text", "openssh:count:x")
        self.assertEqual(ref["id"].split(":", 1)[0], "openssh")


class GoldProvenanceTest(unittest.TestCase):
    def test_deterministic_gold_carries_no_model_block(self):
        provenance = gold_provenance(
            "deterministic_aggregation", "m@v1", "2026-08-01T00:00:00Z", "sha256:ab"
        )
        self.assertNotIn("model", provenance)

    def test_model_assisted_gold_carries_the_model_block(self):
        provenance = gold_provenance(
            "independent_model_then_human",
            "m@v1",
            "2026-08-01T00:00:00Z",
            "sha256:ab",
            model={"name": "x"},
        )
        self.assertEqual(provenance["model"], {"name": "x"})


class SplitSentencesTest(unittest.TestCase):
    def test_splits_on_terminators(self):
        self.assertEqual(
            split_sentences("One. Two! Three?"), ["One.", "Two!", "Three?"]
        )

    def test_drops_empty_fragments(self):
        self.assertEqual(split_sentences("  Only one.  "), ["Only one."])

    def test_empty_answer_yields_no_claims(self):
        self.assertEqual(split_sentences("   "), [])


class LinesFromBytesTest(unittest.TestCase):
    def test_drops_exactly_one_trailing_newline(self):
        self.assertEqual(lines_from_bytes(b"a\nb\n"), ["a", "b"])

    def test_keeps_a_genuine_blank_final_line(self):
        self.assertEqual(lines_from_bytes(b"a\nb\n\n"), ["a", "b", ""])

    def test_handles_no_trailing_newline(self):
        self.assertEqual(lines_from_bytes(b"a\nb"), ["a", "b"])

    def test_empty_input_has_no_lines(self):
        self.assertEqual(lines_from_bytes(b""), [])

    def test_invalid_utf8_is_replaced_not_raised(self):
        self.assertEqual(len(lines_from_bytes(b"a\xffb\n")), 1)


if __name__ == "__main__":
    unittest.main()
