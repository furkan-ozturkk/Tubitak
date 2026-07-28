"""Dataset composition tables from a finished questions file.

Reads what ``main.py --command generate`` wrote and reports what is in it. The
counts come from the same ``src.utils.helper_records`` loader the validator uses,
so a table and a validation report over the same file cannot disagree about how
many records there are.

Usage:
  python3 analysis/analysis_tables.py
  python3 analysis/analysis_tables.py --questions output/pilot/questions.json
  python3 analysis/analysis_tables.py --table dataset
  python3 analysis/analysis_tables.py --questions 'output/pilot/*.json' --table all
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.helper_records import (
    count_by,
    dataset_key_from_evidence,
    load_questions,
)

DEFAULT_QUESTIONS = "output/pilot/questions.json"
TABLES = ("composition", "dataset", "provenance", "all")


def _print_counts(title: str, counts: dict[str, int], total: int) -> None:
    """Prints one count table with a percentage column.

    Args:
        title: Table heading.
        counts: Value-to-count mapping.
        total: Denominator for the percentage column.
    """
    print(f"\n### {title}")
    if not counts:
        print("  (empty)")
        return
    width = max(len(str(k)) for k in counts)
    for key in sorted(counts, key=lambda k: (-counts[k], str(k))):
        share = 100.0 * counts[key] / total if total else 0.0
        print(f"  {str(key):<{width}}  {counts[key]:>5}  {share:>5.1f}%")


def table_composition(records: list[dict[str, Any]]) -> None:
    """Prints the tier / routing / split / review-status breakdown.

    Args:
        records: Loaded question records.
    """
    total = len(records)
    print(f"\n========== COMPOSITION ({total} question(s)) ==========")
    _print_counts("By difficulty", count_by(records, "difficulty"), total)
    _print_counts("By routing path", count_by(records, "routing_path"), total)
    _print_counts("By split", count_by(records, "split"), total)
    _print_counts("By review status", count_by(records, "review_status"), total)
    _print_counts("By task", count_by(records, "task"), total)


def table_dataset(records: list[dict[str, Any]]) -> None:
    """Prints per-LogHub-dataset coverage, split by tier.

    Coverage is what tells a reader whether a corpus-wide claim is supported: a
    dataset contributing only easy questions supports a claim about aggregation
    routing and nothing about synthesis.

    Args:
        records: Loaded question records.
    """
    print(f"\n========== PER DATASET ({len(records)} question(s)) ==========")
    per_dataset: dict[str, dict[str, int]] = {}
    for record in records:
        key = dataset_key_from_evidence(record) or "?"
        tier = record.get("difficulty", "?")
        row = per_dataset.setdefault(
            key, {"easy": 0, "medium": 0, "hard": 0, "total": 0}
        )
        row[tier] = row.get(tier, 0) + 1
        row["total"] += 1

    if not per_dataset:
        print("  (no records carry resolvable evidence ids)")
        return

    width = max(len(k) for k in per_dataset)
    print(f"  {'dataset':<{width}}  {'easy':>5} {'medium':>7} {'hard':>5} {'total':>6}")
    for key in sorted(per_dataset):
        row = per_dataset[key]
        print(
            f"  {key:<{width}}  {row['easy']:>5} {row['medium']:>7} "
            f"{row['hard']:>5} {row['total']:>6}"
        )


def table_provenance(records: list[dict[str, Any]]) -> None:
    """Prints which module and which model produced the gold answers.

    The model column is the Section 5.5/6 integrity claim in a readable form: a
    row whose drafting model equals the reviewing model of any other row would be
    visible here, which no per-record check surfaces.

    Args:
        records: Loaded question records.
    """
    print(f"\n========== PROVENANCE ({len(records)} question(s)) ==========")
    total = len(records)
    methods: dict[str, int] = {}
    created_by: dict[str, int] = {}
    models: dict[str, int] = {}
    for record in records:
        provenance = record.get("gold_provenance", {})
        methods[provenance.get("method", "?")] = (
            methods.get(provenance.get("method", "?"), 0) + 1
        )
        created_by[provenance.get("created_by", "?")] = (
            created_by.get(provenance.get("created_by", "?"), 0) + 1
        )
        model = provenance.get("model")
        name = model.get("name", "?") if model else "(no model)"
        models[name] = models.get(name, 0) + 1

    _print_counts("By method", methods, total)
    _print_counts("By created_by", created_by, total)
    _print_counts("By drafting model", models, total)


def _print_validation_report(report_path: Path) -> None:
    """Prints the verdict from a validation report, when one sits beside the dataset.

    Args:
        report_path: Path to ``validation_report.json``.
    """
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verdict = "PASSED" if report.get("passed") else "FAILED"
    snapshot = report.get("config_snapshot") or {}
    print(f"\n========== VALIDATION REPORT ({report_path}) ==========")
    print(
        f"  {verdict}: {report.get('error_count', '?')} error(s), "
        f"{report.get('warning_count', '?')} warning(s)"
    )
    if snapshot.get("code_version"):
        print(f"  code_version: {snapshot['code_version']}")
    if snapshot.get("gold_draft_model"):
        print(
            f"  models: draft={snapshot.get('gold_draft_model')} "
            f"review={snapshot.get('groundedness_model')}"
        )


def main() -> int:
    """Parses this module's own arguments and prints the requested tables.

    Returns:
        ``0``, or ``2`` when the questions pattern matched nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[DEFAULT_QUESTIONS],
        help="JSON/JSONL file(s) or glob pattern(s) to report on",
    )
    parser.add_argument(
        "--table",
        default="all",
        choices=list(TABLES),
        help="Which table to print",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="validation_report.json to summarise; defaults to the one beside the first questions file",
    )
    args = parser.parse_args()

    records = load_questions(args.questions)
    if not records:
        print(
            f"ERROR: no question records found for {args.questions}.", file=sys.stderr
        )
        return 2

    if args.table in ("composition", "all"):
        table_composition(records)
    if args.table in ("dataset", "all"):
        table_dataset(records)
    if args.table in ("provenance", "all"):
        table_provenance(records)

    report_path = args.report or (
        Path(records[0]["_source_file"]).parent / "validation_report.json"
    )
    _print_validation_report(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
