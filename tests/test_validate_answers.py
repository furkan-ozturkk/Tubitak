"""Tests for the answer layer: does a wrong gold answer actually fail validation?

These are mutation tests. Each one starts from a record that validates and breaks
exactly one thing a reader would care about -- the count, the Yes/No, the looked-up
line, the timestamp format -- and asserts the validator rejects it. Before that layer
existed, every mutation below passed with zero errors, because claims were recomputed
and never compared to the answer they were supposed to support.

Postgres is replaced by a fake repository built from an in-memory corpus. That is the
right substitution for these cases, because what is under test is the comparison
between an answer and a recomputed value, not the database. It is the wrong
substitution for ``check_corpus_matches_database``, whose whole purpose is to detect a
database that disagrees with the file: faking it there would let the fake agree with
itself. That check needs a real loaded Postgres and belongs in an integration suite
against ``docker/compose.yml``, which does not exist yet.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import validate
from src.data.corpus_loader import sha256_bytes, sha256_line
from src.params.validation_params import ValidationConfig
from src.utils.helper_splits import resolve_splits

CORPUS_LINES = [
    "Jul 10 16:01:43 combo sshd: authentication failure for root",
    "Jul 10 16:01:44 combo sshd: authentication failure for admin",
    "Jul 10 16:01:45 combo sshd: session opened for user news",
    "Jul 10 16:01:46 combo sshd: authentication failure for guest",
]
CORPUS_BYTES = ("\n".join(CORPUS_LINES) + "\n").encode("utf-8")
CORPUS_SHA = sha256_bytes(CORPUS_BYTES)
SCHEMA_PATH = Path(REPO_ROOT) / "config" / "question_schema.json"


class FakeCorpusRepository:
    """In-memory stand-in for ``src.utils.helper_postgres``.

    Implements the same three read operations the validator uses, over a list of
    lines, so a test can state a corpus and then mutate a record against it.
    """

    def __init__(self, lines: list[str]):
        """Stores the corpus.

        Args:
            lines: The corpus lines, index ``i`` being line ``i + 1``.
        """
        self.lines = lines

    def get_line(self, dataset: str, line_number: int) -> str:
        """Returns one line's text.

        Args:
            dataset: Ignored; the fake holds one dataset.
            line_number: 1-based line number.

        Returns:
            The line's text.

        Raises:
            RuntimeError: If the line does not exist.
        """
        if not 1 <= line_number <= len(self.lines):
            raise RuntimeError(f"no row for line_number={line_number}")
        return self.lines[line_number - 1]

    def count_literal(
        self, dataset: str, literal: str, case_sensitive: bool = False
    ) -> tuple[int, list[int]]:
        """Counts lines containing a literal.

        Args:
            dataset: Ignored.
            literal: Substring to match.
            case_sensitive: Whether matching is case sensitive.

        Returns:
            Tuple ``(count, line_numbers)``.
        """
        matches = []
        for index, line in enumerate(self.lines, start=1):
            haystack = line if case_sensitive else line.lower()
            needle = literal if case_sensitive else literal.lower()
            if needle in haystack:
                matches.append(index)
        return len(matches), matches

    def count_regex(
        self, dataset: str, pattern: str, case_sensitive: bool = False
    ) -> tuple[int, list[int]]:
        """Counts lines matching a regular expression.

        Args:
            dataset: Ignored.
            pattern: Regular expression.
            case_sensitive: Whether matching is case sensitive.

        Returns:
            Tuple ``(count, line_numbers)``.
        """
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
        matches = [
            index
            for index, line in enumerate(self.lines, start=1)
            if compiled.search(line)
        ]
        return len(matches), matches

    def fetch_all_lines(self, dataset: str) -> list[tuple[int, str]]:
        """Returns every line with its number.

        Args:
            dataset: Ignored.

        Returns:
            ``(line_number, text)`` pairs.
        """
        return list(enumerate(self.lines, start=1))


def count_record() -> dict:
    """Builds a valid count record over the test corpus.

    Returns:
        A record whose answer, claim and evidence all agree.
    """
    literal = "authentication failure"
    group_id = "linux:count:authentication_failure"
    line_numbers = [1, 2, 4]
    return {
        "id": "linux_v1_count_authentication_failure_0",
        "question": "How many log lines contain 'authentication failure'?",
        "routing_path": "sql",
        "answer_type": "count",
        "task": "Aggregation",
        "difficulty": "easy",
        "phrasing_family": "how-many",
        "split": "dev",
        "review_status": "verified",
        "reviewers": ["faz1_pilot_script"],
        "expected_answer": "3",
        "gold_provenance": {
            "method": "deterministic_aggregation",
            "created_by": "src/generators/easy_tier.py@v1",
            "created_at": "2026-08-01T00:00:00Z",
            "corpus_sha256": CORPUS_SHA,
        },
        "numeric_claims": [
            {
                "value": 3,
                "query": {
                    "operator": "count_literal",
                    "literal": literal,
                    "case_sensitive": False,
                },
            }
        ],
        "evidence": {
            "refs": [
                {
                    "id": f"linux:line:{number:08d}:{sha256_line(CORPUS_LINES[number - 1])[7:23]}",
                    "line_number": number,
                    "line_hash": sha256_line(CORPUS_LINES[number - 1]),
                    "group_id": group_id,
                }
                for number in line_numbers
            ]
        },
    }


def presence_record() -> dict:
    """Builds a valid presence record over the test corpus.

    Returns:
        A record answering "No" to a literal the corpus does not contain.
    """
    group_id = "linux:presence:invalid_user"
    return {
        "id": "linux_v1_presence_invalid_user_0",
        "question": "Does the log contain any line with 'Invalid user'?",
        "routing_path": "sql",
        "answer_type": "presence",
        "task": "Aggregation",
        "difficulty": "easy",
        "phrasing_family": "how-many",
        "split": "dev",
        "review_status": "verified",
        "reviewers": ["faz1_pilot_script"],
        "expected_answer": "No",
        "gold_provenance": {
            "method": "deterministic_aggregation",
            "created_by": "src/generators/easy_tier.py@v1",
            "created_at": "2026-08-01T00:00:00Z",
            "corpus_sha256": CORPUS_SHA,
        },
        "numeric_claims": [
            {
                "value": 0,
                "query": {
                    "operator": "count_literal",
                    "literal": "Invalid user",
                    "case_sensitive": False,
                },
            }
        ],
        "evidence": {
            "refs": [
                {
                    "id": f"linux:line:00000001:{sha256_line(CORPUS_LINES[0])[7:23]}",
                    "line_number": 1,
                    "line_hash": sha256_line(CORPUS_LINES[0]),
                    "group_id": group_id,
                }
            ]
        },
    }


def lookup_record() -> dict:
    """Builds a valid line-lookup record over the test corpus.

    Returns:
        A record whose answer is exactly the text of the line it cites.
    """
    group_id = "linux:lookup:authentication_failure_first"
    return {
        "id": "linux_v1_lookup_authentication_failure_first_0",
        "question": "Show the first line that reports 'authentication failure'.",
        "routing_path": "keyword",
        "answer_type": "line_lookup",
        "task": "Lookup",
        "difficulty": "easy",
        "phrasing_family": "how-many",
        "split": "dev",
        "review_status": "verified",
        "reviewers": ["faz1_pilot_script"],
        "expected_answer": CORPUS_LINES[0],
        "gold_provenance": {
            "method": "deterministic_aggregation",
            "created_by": "src/generators/easy_tier.py@v1",
            "created_at": "2026-08-01T00:00:00Z",
            "corpus_sha256": CORPUS_SHA,
        },
        "evidence": {
            "refs": [
                {
                    "id": f"linux:line:00000001:{sha256_line(CORPUS_LINES[0])[7:23]}",
                    "line_number": 1,
                    "line_hash": sha256_line(CORPUS_LINES[0]),
                    "group_id": group_id,
                }
            ]
        },
    }


class ValidatorAnswerTest(unittest.TestCase):
    """Runs the validator over in-memory records with a fake corpus repository."""

    def setUp(self):
        self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.repository = FakeCorpusRepository(CORPUS_LINES)
        self.corpus_index = {
            "linux": {
                "path": Path("Linux_2k.log"),
                "sha256": CORPUS_SHA,
                "lines": CORPUS_LINES,
            }
        }
        self.config = ValidationConfig(
            questions=("memory",), schema=SCHEMA_PATH, corpus_dir=Path("memory")
        )

    def run_validator(self, records: list[dict]) -> tuple[list[str], list[str]]:
        """Validates records against the fake repository.

        Args:
            records: Records to validate. Splits are resolved first, so a test only
                fails on the thing it mutated.

        Returns:
            Tuple ``(errors, warnings)``.
        """
        for record in records:
            record["_source_file"] = "memory"
        resolve_splits(records)
        with mock.patch.object(validate, "helper_postgres", self.repository):
            errors, warnings, _stats = validate.validate_records(
                records, self.schema, self.corpus_index, self.config
            )
        return errors, warnings

    def assertRejects(self, records: list[dict], needle: str) -> None:
        """Asserts validation failed with an error mentioning ``needle``.

        Args:
            records: Records to validate.
            needle: Substring the failure must mention.
        """
        errors, _warnings = self.run_validator(records)
        self.assertTrue(errors, "expected at least one error, got none")
        self.assertTrue(
            any(needle in error for error in errors),
            f"no error mentioned {needle!r}; errors were {errors}",
        )

    def test_valid_records_pass(self):
        errors, _warnings = self.run_validator(
            [count_record(), presence_record(), lookup_record()]
        )
        self.assertEqual(errors, [])

    def test_count_answer_disagreeing_with_recomputed_value_fails(self):
        record = count_record()
        record["expected_answer"] = "999999"
        self.assertRejects([record], "does not match the recomputed count")

    def test_count_claim_disagreeing_with_corpus_fails(self):
        record = count_record()
        record["numeric_claims"][0]["value"] = 490
        record["expected_answer"] = "490"
        self.assertRejects([record], "could not be reproduced")

    def test_presence_no_with_positive_count_fails(self):
        record = presence_record()
        record["numeric_claims"][0]["query"]["literal"] = "authentication failure"
        record["numeric_claims"][0]["value"] = 3
        self.assertRejects([record], "contradicts the recomputed count")

    def test_presence_yes_with_zero_count_fails(self):
        record = presence_record()
        record["expected_answer"] = "Yes"
        self.assertRejects([record], "contradicts the recomputed count")

    def test_lookup_answer_not_matching_cited_line_fails(self):
        record = lookup_record()
        record["expected_answer"] = CORPUS_LINES[2]
        self.assertRejects([record], "does not equal any cited line")

    def test_invalid_created_at_fails_format_checking(self):
        record = count_record()
        record["gold_provenance"]["created_at"] = "not-a-date"
        self.assertRejects([record], "SCHEMA ERROR")

    def test_evidence_hash_mismatch_fails(self):
        record = count_record()
        record["evidence"]["refs"][0]["line_hash"] = sha256_line("something else")
        self.assertRejects([record], "line_hash mismatch")

    def test_corpus_sha_mismatch_fails(self):
        record = count_record()
        record["gold_provenance"]["corpus_sha256"] = sha256_bytes(b"other corpus")
        self.assertRejects([record], "corpus_sha256 does not match")

    def test_duplicate_ids_fail(self):
        self.assertRejects([count_record(), count_record()], "duplicate id")

    def test_wrong_stored_split_fails(self):
        record = count_record()
        record["_source_file"] = "memory"
        records = [record]
        resolve_splits(records)
        record["split"] = "test" if record["split"] == "dev" else "dev"
        with mock.patch.object(validate, "helper_postgres", self.repository):
            errors, _warnings, _stats = validate.validate_records(
                records, self.schema, self.corpus_index, self.config
            )
        self.assertTrue(
            any("event set assigned" in error for error in errors),
            f"expected a split error, got {errors}",
        )

    def test_missing_corpus_is_an_error_not_a_silent_pass(self):
        record = count_record()
        record["_source_file"] = "memory"
        records = [record]
        resolve_splits(records)
        with mock.patch.object(validate, "helper_postgres", self.repository):
            errors, _warnings, _stats = validate.validate_records(
                records, self.schema, {}, self.config
            )
        self.assertTrue(
            any("corpus checks did not run" in error for error in errors),
            f"expected a missing-corpus error, got {errors}",
        )

    def test_model_drafted_record_cannot_be_verified_without_a_human(self):
        record = count_record()
        record["difficulty"] = "medium"
        record["routing_path"] = "semantic"
        record["answer_type"] = "explanation"
        record["expected_answer"] = "Repeated SSH authentication failures."
        record.pop("numeric_claims")
        record["review_status"] = "verified"
        record["reviewers"] = ["faz1_pilot_script"]
        record["gold_provenance"]["method"] = "independent_model_then_human"
        record["gold_provenance"]["model"] = {
            "name": "nemotron-3-nano:30b",
            "family": "nemotron",
            "digest": "sha256:abc",
            "prompt_sha256": sha256_line("prompt"),
        }
        self.assertRejects([record], "names no human")


if __name__ == "__main__":
    unittest.main()
