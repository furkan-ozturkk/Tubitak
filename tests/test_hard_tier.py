"""Tests for the hard tier's group-selection and evidence-selection algorithms.

Synthetic corpora rather than a fetched LogHub file: these test the algorithms
in isolation (determinism, the proven-link requirement, salience scoring), not
whether any particular dataset's regex happens to match well -- that is a
curation question checked by hand against the real corpus, not something a
portable unit test should depend on.
"""

import os
import sys
import unittest
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dataset_specs import HardComparativeSpec, HardCorrelationSpec
from src.generators.hard_tier import (
    _build_evidence_block,
    _select_correlation_sets,
    _select_group_sets,
    _select_salient_indices,
)


def _view(lines: list[str]) -> SimpleNamespace:
    """Builds a minimal stand-in for ``CorpusView`` (only ``.lines``/``.key`` used)."""
    return SimpleNamespace(lines=lines, key="test")


class SelectGroupSetsTest(unittest.TestCase):
    def setUp(self):
        self.spec = HardComparativeSpec(
            spec_id="test_compare",
            task="Comparison",
            extract_key_regex=r"(?P<key>node[A-Z])",
            min_lines_per_group=3,
            num_groups=2,
            question_templates=("Compare {key0} and {key1}.",),
        )

    def test_picks_the_two_largest_qualifying_groups(self):
        lines = ["nodeA event"] * 5 + ["nodeB event"] * 4 + ["nodeC event"] * 3
        sets = _select_group_sets(lines, self.spec, max_sets=5)
        self.assertEqual(len(sets), 1)
        keys = [key for key, _ in sets[0]]
        self.assertEqual(keys, ["nodeA", "nodeB"])

    def test_groups_below_the_minimum_are_dropped(self):
        lines = ["nodeA event"] * 5 + ["nodeB event"] * 2
        sets = _select_group_sets(lines, self.spec, max_sets=5)
        self.assertEqual(sets, [])

    def test_chunking_never_reuses_an_entity(self):
        lines = (
            ["nodeA e"] * 6
            + ["nodeB e"] * 5
            + ["nodeC e"] * 4
            + ["nodeD e"] * 3
        )
        sets = _select_group_sets(lines, self.spec, max_sets=5)
        seen = set()
        for pair in sets:
            for key, _indices in pair:
                self.assertNotIn(key, seen)
                seen.add(key)

    def test_is_deterministic_across_runs(self):
        lines = ["nodeA e"] * 5 + ["nodeB e"] * 5 + ["nodeC e"] * 5
        first = _select_group_sets(lines, self.spec, max_sets=5)
        second = _select_group_sets(lines, self.spec, max_sets=5)
        self.assertEqual(first, second)


class SelectCorrelationSetsTest(unittest.TestCase):
    def setUp(self):
        self.spec = HardCorrelationSpec(
            spec_id="test_correlate",
            task="Correlation",
            key_a_regex=r"(?P<key>container_[0-9]+)",
            key_b_regex=r"(?P<key>attempt_[0-9]+)",
            min_lines_per_group=3,
            question_templates=("How does {key0} relate to {key1}?",),
        )

    def test_requires_a_line_naming_both_entities_together(self):
        lines = (
            ["container_1 launched"] * 3
            + ["attempt_1 started"] * 3
            + ["container_2 launched"] * 3
            + ["attempt_2 started"] * 3
        )
        sets = _select_correlation_sets(lines, self.spec, max_pairs=5)
        self.assertEqual(sets, [], "no line names both ids, so nothing is proven")

    def test_finds_a_proven_pair_and_collects_every_matching_line(self):
        lines = [
            "container_1 launched for attempt_1",
            "container_1 running",
            "container_1 finished",
            "attempt_1 progress 10%",
            "attempt_1 progress 50%",
        ]
        sets = _select_correlation_sets(lines, self.spec, max_pairs=5)
        self.assertEqual(len(sets), 1)
        value_a, indices_a, value_b, indices_b = sets[0]
        self.assertEqual(value_a, "container_1")
        self.assertEqual(value_b, "attempt_1")
        self.assertEqual(len(indices_a), 3)
        self.assertEqual(len(indices_b), 3)

    def test_pairs_below_the_minimum_on_either_side_are_dropped(self):
        lines = [
            "container_1 launched for attempt_1",
            "attempt_1 progress 10%",
        ]
        sets = _select_correlation_sets(lines, self.spec, max_pairs=5)
        self.assertEqual(sets, [])

    def test_no_entity_is_reused_across_pairs(self):
        lines = [
            "container_1 launched for attempt_1",
            "container_1 running",
            "container_1 finished",
            "attempt_1 progress 10%",
            "attempt_1 progress 50%",
            "container_2 launched for attempt_1",
            "container_2 running",
            "container_2 finished",
        ]
        sets = _select_correlation_sets(lines, self.spec, max_pairs=5)
        seen_b = set()
        for _value_a, _indices_a, value_b, _indices_b in sets:
            self.assertNotIn(value_b, seen_b)
            seen_b.add(value_b)


class SelectSalientIndicesTest(unittest.TestCase):
    def test_returns_everything_when_already_within_cap(self):
        lines = ["line %d" % i for i in range(5)]
        view = _view(lines)
        kept = _select_salient_indices(view, list(range(5)), cap=10)
        self.assertEqual(kept, list(range(5)))

    def test_always_keeps_the_true_first_and_last_line(self):
        lines = ["plain %d" % i for i in range(20)]
        view = _view(lines)
        kept = _select_salient_indices(view, list(range(20)), cap=5)
        self.assertIn(0, kept)
        self.assertIn(19, kept)
        self.assertEqual(len(kept), 5)

    def test_prefers_lines_with_a_salience_keyword_over_plain_repeats(self):
        lines = ["plain line %d" % i for i in range(10)]
        lines[4] = "a retry occurred here"
        lines[7] = "recovered successfully afterwards"
        view = _view(lines)
        kept = _select_salient_indices(view, list(range(10)), cap=4)
        self.assertIn(4, kept)
        self.assertIn(7, kept)

    def test_result_is_sorted_ascending(self):
        lines = ["plain %d" % i for i in range(10)]
        lines[6] = "an error happened"
        view = _view(lines)
        kept = _select_salient_indices(view, list(range(10)), cap=4)
        self.assertEqual(kept, sorted(kept))


class BuildEvidenceBlockTest(unittest.TestCase):
    def test_marks_a_gap_between_non_adjacent_kept_lines(self):
        lines = ["plain %d" % i for i in range(10)]
        lines[5] = "a timeout was retried"
        view = _view(lines)
        refs, block = _build_evidence_block(view, "k", list(range(10)), cap=3, group_id="g")
        self.assertEqual(len(refs), 3)
        self.assertIn("more lines follow between these", block)

    def test_cites_every_line_without_a_gap_marker_when_under_cap(self):
        lines = ["plain %d" % i for i in range(4)]
        view = _view(lines)
        refs, block = _build_evidence_block(view, "k", list(range(4)), cap=10, group_id="g")
        self.assertEqual(len(refs), 4)
        self.assertNotIn("more lines follow between these", block)


if __name__ == "__main__":
    unittest.main()
