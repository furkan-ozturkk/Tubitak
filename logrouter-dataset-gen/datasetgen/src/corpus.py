#!/usr/bin/env python3
"""
corpus.py

Shared corpus-loading and hashing helpers so every question generator
computes line numbers and evidence hashes exactly the way
schema/validate_schema.py re-derives and checks them at verification time.
"""
import hashlib
from pathlib import Path


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
    """Lowercase dataset name, used as the evidence-id / lookup prefix (matches validate_schema.py)."""
    return name.lower()
