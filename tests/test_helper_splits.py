"""Tests for leak-proof dev/test split assignment."""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.helper_splits import (
    expected_splits,
    group_ids_of,
    resolve_splits,
    split_for_component,
)


def record(*group_ids: str) -> dict:
    """Builds a minimal record citing one line per group.

    The refs deliberately carry no ``id``, so these synthetic records exercise
    only the co-citation edges; ``lined_record`` below builds refs that also
    participate in line-sharing edges.

    Args:
        *group_ids: Evidence group ids the record cites.

    Returns:
        A record with just the fields split assignment reads.
    """
    return {
        "evidence": {
            "refs": [
                {"group_id": group_id, "line_number": index + 1}
                for index, group_id in enumerate(group_ids)
            ]
        }
    }


def lined_record(group_id: str, dataset: str, *line_numbers: int) -> dict:
    """Builds a record whose refs carry real-looking ids for line-edge tests.

    Args:
        group_id: The single evidence group the record cites.
        dataset: Dataset key embedded in each ref id.
        *line_numbers: The 1-based lines the record cites.

    Returns:
        A record with just the fields split assignment reads.
    """
    return {
        "evidence": {
            "refs": [
                {
                    "group_id": group_id,
                    "line_number": line_number,
                    "id": f"{dataset}:line:{line_number:08d}:deadbeefdeadbeef",
                }
                for line_number in line_numbers
            ]
        }
    }


class GroupIdsOfTest(unittest.TestCase):
    def test_returns_distinct_group_ids(self):
        self.assertEqual(group_ids_of(record("a", "b", "a")), frozenset({"a", "b"}))

    def test_record_without_evidence_has_no_groups(self):
        self.assertEqual(group_ids_of({}), frozenset())


class SplitForComponentTest(unittest.TestCase):
    def test_is_deterministic(self):
        self.assertEqual(
            split_for_component("hdfs:count:x"), split_for_component("hdfs:count:x")
        )

    def test_zero_fraction_sends_everything_to_dev(self):
        for name in ("a", "b", "c", "d", "e"):
            self.assertEqual(split_for_component(name, test_fraction=0.0), "dev")

    def test_full_fraction_is_rejected_by_args_not_here(self):
        self.assertIn(split_for_component("a", test_fraction=0.999), ("dev", "test"))


class ExpectedSplitsTest(unittest.TestCase):
    def test_co_cited_groups_share_a_split(self):
        records = [record("g1"), record("g2"), record("g1", "g2")]
        splits = expected_splits(records)
        self.assertEqual(len(set(splits.values())), 1)

    def test_transitive_linking_merges_components(self):
        records = [record("a", "b"), record("b", "c"), record("c"), record("a")]
        splits = expected_splits(records)
        self.assertEqual(len(set(splits.values())), 1)

    def test_unlinked_groups_may_differ(self):
        records = [record(f"group-{index}") for index in range(40)]
        splits = expected_splits(records)
        self.assertEqual(set(splits.values()), {"dev", "test"})

    def test_single_group_record_matches_bare_group_hash(self):
        records = [record("hdfs:count:packetresponder")]
        self.assertEqual(
            expected_splits(records)[0],
            split_for_component("hdfs:count:packetresponder"),
        )

    def test_records_without_evidence_are_absent(self):
        self.assertEqual(expected_splits([{}]), {})

    def test_order_does_not_change_the_assignment(self):
        forward = [record("a", "b"), record("c"), record("b")]
        backward = list(reversed(forward))
        self.assertEqual(
            set(expected_splits(forward).values()),
            set(expected_splits(backward).values()),
        )


class LineSharingTest(unittest.TestCase):
    def test_groups_sharing_a_line_share_a_split(self):
        records = [
            lined_record("linux:count:auth", "linux", 3, 4, 5),
            lined_record("linux:semantic:auth_0", "linux", 5, 6, 7),
        ]
        splits = expected_splits(records)
        self.assertEqual(splits[0], splits[1])

    def test_line_sharing_is_transitive(self):
        records = [
            lined_record("g:a", "bgl", 1, 2),
            lined_record("g:b", "bgl", 2, 3),
            lined_record("g:c", "bgl", 3, 4),
        ]
        splits = expected_splits(records)
        self.assertEqual(len(set(splits.values())), 1)

    def test_same_line_number_in_different_datasets_does_not_link(self):
        records = [
            lined_record("linux:count:x", "linux", 1),
            lined_record("mac:count:y", "mac", 1),
        ]
        splits = expected_splits(records)
        self.assertEqual(splits[0], expected_splits([records[0]])[0])
        self.assertEqual(splits[1], expected_splits([records[1]])[0])

    def test_disjoint_lines_do_not_link(self):
        records = [
            lined_record("linux:count:x", "linux", 1, 2),
            lined_record("linux:semantic:y", "linux", 10, 11),
        ]
        splits = expected_splits(records)
        self.assertEqual(splits[0], split_for_component("linux:count:x"))
        self.assertEqual(splits[1], split_for_component("linux:semantic:y"))

    def test_refs_without_ids_never_line_link(self):
        records = [record("a"), record("b")]
        splits = expected_splits(records)
        self.assertEqual(splits[0], split_for_component("a"))
        self.assertEqual(splits[1], split_for_component("b"))


class ResolveSplitsTest(unittest.TestCase):
    def test_stamps_every_record(self):
        records = [record("a"), record("a", "b")]
        resolve_splits(records)
        self.assertEqual(records[0]["split"], records[1]["split"])

    def test_overwrites_a_wrong_stored_split(self):
        records = [record("a"), record("a", "b")]
        records[0]["split"] = "dev"
        records[1]["split"] = "test"
        resolve_splits(records)
        self.assertEqual(records[0]["split"], records[1]["split"])


if __name__ == "__main__":
    unittest.main()
