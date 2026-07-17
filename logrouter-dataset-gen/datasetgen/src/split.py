#!/usr/bin/env python3
"""
split.py

Deterministic dev/test split assignment (Section 6): every question whose
evidence.refs share a group_id must land in the same split, so the split is
derived purely from a hash of group_id rather than being assigned randomly
per record. This guarantees the same group_id always maps to the same
split across repeated runs (determinism, Section 2).
"""
import hashlib

TEST_FRACTION = 0.20


def split_for_group(group_id: str, test_fraction: float = TEST_FRACTION) -> str:
    digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "dev"
