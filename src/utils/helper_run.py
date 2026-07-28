"""Run-level plumbing: writing outputs and printing what a run did.

Every file this project writes goes through ``write_json``, and it writes to a
temporary file in the destination directory before ``os.replace`` moves it into
place. That matters more here than in most projects: the dataset file is the
deliverable, ``review-apply`` rewrites it in place, and a process killed
mid-``write_text`` would leave a truncated JSON array where the dataset used to be
with no copy of what it replaced. ``os.replace`` is atomic on the same filesystem,
so a reader sees either the old file or the new one.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.params.results_params import GenerationSummary, ValidationStats


def write_json(path: Path, payload: Any) -> None:
    """Writes a JSON document atomically, creating parent directories as needed.

    Args:
        path: Destination file.
        payload: Any JSON-serialisable object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def write_text(path: Path, text: str) -> None:
    """Writes a text file atomically, creating parent directories as needed.

    Args:
        path: Destination file.
        text: Contents to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as temp_file:
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def print_generation_summary(summary: GenerationSummary) -> None:
    """Prints the per-tier counts from one generate pass.

    The configured target and mix are printed beside the realised counts as
    context, not as a verdict. Nothing enforces the mix: the tiers produce what the
    curated specs and the corpus allow, and hitting a ratio by discarding valid
    questions would be worse than reporting the gap (Section 3.2). The realised
    shares are printed too, so the gap is visible rather than implied.

    Args:
        summary: Counts and configuration from the pass.
    """
    print("\n=== SUMMARY ===")
    label = "pilot dataset" if summary.official_set else "--full scratch pass"
    total = summary.total
    print(
        f"easy={summary.easy} medium={summary.medium} hard={summary.hard} "
        f"total={total} ({label})"
    )
    if total and summary.difficulty_mix is not None:
        realised = (
            f"easy={summary.easy / total:.2f} "
            f"medium={summary.medium / total:.2f} "
            f"hard={summary.hard / total:.2f}"
        )
        configured = (
            f"easy={summary.difficulty_mix.easy:.2f} "
            f"medium={summary.difficulty_mix.medium:.2f} "
            f"hard={summary.difficulty_mix.hard:.2f}"
        )
        print(f"realised mix : {realised}")
        print(f"target mix   : {configured} (reported, not enforced)")
    if summary.target_total is not None:
        print(f"target total : {summary.target_total} (reported, not enforced)")
    print(f"Wrote dataset to {summary.out}")


def print_validation_stats(stats: ValidationStats) -> None:
    """Prints the composition of a validated dataset.

    Args:
        stats: Stats collected during validation.
    """
    print("=== STATS ===")
    for key, value in stats.as_dict().items():
        print(f"  {key}: {value}")


def print_findings(errors: list[str], warnings: list[str], limit: int = 50) -> None:
    """Prints validation warnings and errors, truncated to a readable length.

    A failing run can produce one finding per record, and a several-hundred-line
    dump buries the first one, which is usually the one that explains the rest. The
    full lists always reach the JSON report, so nothing truncated here is lost.

    Args:
        errors: Failures. Printed to stderr.
        warnings: Findings that are expectations rather than rules.
        limit: How many of each to print before summarising the remainder.
    """
    if warnings:
        print(f"\n=== WARNINGS ({len(warnings)}) ===")
        for warning in warnings[:limit]:
            print(f"  - {warning}")
        if len(warnings) > limit:
            print(f"  ... (+{len(warnings) - limit} more)")

    if errors:
        print(f"\n=== ERRORS ({len(errors)}) ===", file=sys.stderr)
        for error in errors[:limit]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > limit:
            print(f"  ... (+{len(errors) - limit} more)", file=sys.stderr)
