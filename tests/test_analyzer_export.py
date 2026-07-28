"""Tests for the analyzer-format export.

The export is pure translation: group by dataset, name the corpus file from the
manifest, filter script identities out of ``reviewers``. Each of those is a
place where a wrong answer would silently break the consumer's integrity gate,
so each is asserted directly on ``build_payload``.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.commands.analyzer_export import _human_reviewers, build_payload

MANIFEST = {
    "source": {"pinned_commit": "d" * 40},
    "datasets": [
        {"name": "Linux", "local_filename": "Linux_2k.log"},
        {"name": "HDFS", "local_filename": "HDFS_2k.log"},
    ],
}


def record(
    record_id: str,
    dataset: str,
    corpus_sha: str = "sha256:" + "a" * 64,
    reviewers: list | None = None,
) -> dict:
    """Builds a minimal exportable record.

    Args:
        record_id: The record id.
        dataset: Dataset key embedded in the evidence ref id.
        corpus_sha: ``gold_provenance.corpus_sha256`` value.
        reviewers: ``reviewers`` value; defaults to the generator stamp.

    Returns:
        A record with the fields the export reads.
    """
    return {
        "id": record_id,
        "reviewers": ["faz1_pilot_script"] if reviewers is None else reviewers,
        "gold_provenance": {"corpus_sha256": corpus_sha},
        "evidence": {
            "refs": [
                {
                    "id": f"{dataset}:line:00000001:deadbeefdeadbeef",
                    "line_number": 1,
                    "group_id": f"{dataset}:count:x",
                }
            ]
        },
        "_source_file": "/tmp/in.json",
    }


class BuildPayloadTest(unittest.TestCase):
    def test_groups_records_by_dataset(self):
        payload = build_payload(
            [record("a", "linux"), record("b", "hdfs"), record("c", "linux")],
            MANIFEST,
        )
        self.assertEqual(sorted(payload), ["hdfs", "linux"])
        self.assertEqual(len(payload["linux"]["questions"]), 2)
        self.assertEqual(len(payload["hdfs"]["questions"]), 1)

    def test_log_file_follows_the_analyzer_layout(self):
        payload = build_payload([record("a", "linux")], MANIFEST)
        self.assertEqual(
            payload["linux"]["log_file"], "data/corpus/loghub-2k/Linux/Linux_2k.log"
        )
        self.assertEqual(payload["linux"]["corpus_version"], "d" * 40)

    def test_source_file_annotation_is_stripped(self):
        payload = build_payload([record("a", "linux")], MANIFEST)
        self.assertNotIn("_source_file", payload["linux"]["questions"][0])

    def test_generator_identity_leaves_reviewers(self):
        payload = build_payload([record("a", "linux")], MANIFEST)
        self.assertEqual(payload["linux"]["questions"][0]["reviewers"], [])

    def test_human_reviewers_survive_the_filter(self):
        reviewers = ["faz1_pilot_script", "ada"]
        payload = build_payload([record("a", "linux", reviewers=reviewers)], MANIFEST)
        self.assertEqual(payload["linux"]["questions"][0]["reviewers"], ["ada"])

    def test_unpinned_dataset_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build_payload([record("a", "mac")], MANIFEST)
        self.assertIn("does not pin", str(caught.exception))

    def test_record_without_evidence_is_refused(self):
        bare = {
            "id": "a",
            "reviewers": [],
            "gold_provenance": {},
            "evidence": {"refs": []},
        }
        with self.assertRaises(ValueError):
            build_payload([bare], MANIFEST)

    def test_mixed_corpus_hashes_within_one_dataset_are_refused(self):
        records = [
            record("a", "linux", corpus_sha="sha256:" + "a" * 64),
            record("b", "linux", corpus_sha="sha256:" + "b" * 64),
        ]
        with self.assertRaises(ValueError) as caught:
            build_payload(records, MANIFEST)
        self.assertIn("two corpora", str(caught.exception))

    def test_export_does_not_mutate_the_input_records(self):
        source = record("a", "linux")
        build_payload([source], MANIFEST)
        self.assertIn("_source_file", source)
        self.assertEqual(source["reviewers"], ["faz1_pilot_script"])


class HumanReviewersTest(unittest.TestCase):
    def test_script_stamp_alone_maps_to_empty(self):
        self.assertEqual(_human_reviewers({"reviewers": ["faz1_pilot_script"]}), [])

    def test_missing_reviewers_maps_to_empty(self):
        self.assertEqual(_human_reviewers({}), [])


if __name__ == "__main__":
    unittest.main()
