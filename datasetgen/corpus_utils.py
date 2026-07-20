#!/usr/bin/env python3
"""
corpus_utils.py

Shared corpus-loading, hashing, and split-assignment helpers so every
question generator (question_generators.py) computes line numbers, evidence
hashes, and dev/test splits exactly the way schema_validator.py re-derives
and checks them at verification time.
"""
import hashlib
from pathlib import Path

TEST_FRACTION = 0.20


def load_lines(log_path: Path) -> list:
    data = log_path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def sha256_line(line: str) -> str:
    return "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def dataset_key(name: str) -> str:
    """Lowercase dataset name, used as the evidence-id / lookup prefix (matches schema_validator.py)."""
    return name.lower()


def split_for_group(group_id: str, test_fraction: float = TEST_FRACTION) -> str:
    """
    Deterministic dev/test split assignment (Section 6): every question whose
    evidence.refs share a group_id must land in the same split, so the split is
    derived purely from a hash of group_id rather than being assigned randomly
    per record. This guarantees the same group_id always maps to the same
    split across repeated runs (determinism, Section 2).
    """
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "dev"
