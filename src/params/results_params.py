"""Result dataclasses for what a run produces, and the provenance recorded beside it.

Two outputs, two dataclasses: ``GenerationSummary`` is what ``generate`` prints
at the end of a pass, ``ValidationReport`` is what ``validate`` writes to
``validation_report.json``.

The config snapshot deliberately lives in the *report*, never in a question
record. ``config/question_schema.json`` sets ``additionalProperties: false``, so
a record carries exactly the thirteen fields Section 6 defines and nothing else;
a snapshot smuggled into one would make the dataset fail its own validator. The
report is not schema-constrained, which makes it the right place to answer "which
code and which flags produced the file this report certifies" — a question this
project has already needed answered once, when result files outlived the fixes
that invalidated them.
"""

import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationSummary:
    """Per-tier counts from one ``generate`` pass.

    ``target_total`` and ``difficulty_mix`` come from ``scale_config.yaml`` and
    are recorded next to the realised counts rather than checked against them:
    the mix is a scaling target for Phase 2/3, and a pilot pass that falls short
    of it is a fact to report, not an error to raise (Section 3.2).

    Attributes:
        easy: Easy-tier records produced.
        medium: Medium-tier records produced.
        hard: Hard-tier records produced.
        out: Dataset file written.
        official_set: True when this pass wrote the pilot dataset (the default
            output path) rather than a ``--full`` pass's scratch file. Both run
            the same three tiers at full width; the flag records which file the
            run was aimed at.
        target_total: The run's ``--target_total_questions`` reporting target.
        difficulty_mix: The configured mix, recorded beside the realised counts.
        easy_target_total: The run's ``--easy_target_total``, or ``None`` when
            the model-SQL question source was off and ``easy`` is the curated
            count alone.
    """

    easy: int = 0
    medium: int = 0
    hard: int = 0
    out: Path | None = None
    official_set: bool = False
    target_total: int | None = None
    difficulty_mix: Any | None = None
    easy_target_total: int | None = None

    @property
    def total(self) -> int:
        """Returns the number of records written across all tiers."""
        return self.easy + self.medium + self.hard


@dataclass
class ValidationStats:
    """Composition of the validated dataset.

    Attributes:
        total_questions: Records loaded.
        unique_ids: Distinct ``id`` values; a shortfall means duplicates, which
            are also reported individually as errors.
        distinct_group_ids: Distinct evidence ``group_id`` values.
        by_difficulty: Record count per difficulty tier.
        by_routing_path: Record count per routing path.
        by_split: Record count per dev/test split.
        by_review_status: Record count per review status.
        hard_question_count: Records at ``difficulty=hard``.
        datasets_covered: LogHub dataset keys at least one record cites. Recorded
            because a clean report over one dataset says nothing about the other
            nine, and the printed verdict alone does not reveal which it covered.
    """

    total_questions: int = 0
    unique_ids: int = 0
    distinct_group_ids: int = 0
    by_difficulty: dict[str, int] = field(default_factory=dict)
    by_routing_path: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    by_review_status: dict[str, int] = field(default_factory=dict)
    hard_question_count: int = 0
    datasets_covered: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Returns the stats as a plain dict for JSON serialisation."""
        return asdict(self)


@dataclass
class ValidationReport:
    """Top-level container for one ``validate`` pass.

    Attributes:
        passed: True when no errors were raised. Warnings do not affect it.
        stats: Dataset composition.
        errors: Failures. Any one of these makes the run fail.
        warnings: Findings that Section 2/6 states as expectations rather than
            rules, promoted to errors by ``--strict``.
        config_snapshot: The flags and code version this validation ran under.
    """

    passed: bool = False
    stats: ValidationStats = field(default_factory=ValidationStats)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Returns the report in the JSON shape written to disk."""
        return {
            "passed": self.passed,
            "stats": self.stats.as_dict(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "config_snapshot": self.config_snapshot,
        }


def _code_version() -> str | None:
    """Return the git commit that produced this run, with a dirty marker.

    Recorded beside the numbers rather than folded into any identifier: this is
    provenance the reader can check, and it must not change which file a run
    owns. A dataset generated from a dirty tree is still a valid dataset — it
    just cannot be reproduced from a commit, and that is exactly what the
    ``-dirty`` suffix says.

    Returns:
        The short commit hash, suffixed ``-dirty`` when the working tree has
        uncommitted changes, or ``None`` when git is unavailable or this is not
        a repository (which is the normal case inside the container).
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sha = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return f"{sha}-dirty" if dirty else sha


def build_config_snapshot(args: Any) -> dict[str, Any]:
    """Builds the config_snapshot dict recorded in the validation report.

    Everything a reader would need to re-run this validation is here: the
    command, the paths, the strictness, the curation knobs that shaped the
    records being checked, and the commit. The two model names are included
    because they are the Section 5.5/6 integrity claim in machine-readable
    form — a report asserting a dataset passed is only meaningful alongside
    evidence that its gold answers were not self-certified.

    Args:
        args: Parsed CLI namespace.

    Returns:
        A JSON-serialisable snapshot of the run's configuration.
    """
    return {
        "command": args.command,
        "code_version": _code_version(),
        "experiment_tag": args.experiment_tag,
        "dataset": str(args.dataset),
        "questions": [str(q) for q in args.questions],
        "corpus_dir": str(args.corpus_dir),
        "schema": str(args.schema),
        "manifest": str(args.manifest) if args.manifest else None,
        "strict": args.strict,
        "full": args.full,
        "gold_draft_model": args.gold_draft_model,
        "groundedness_model": args.groundedness_model,
        "test_fraction": args.test_fraction,
        "min_matches": args.min_matches,
        "max_cited_lines": args.max_cited_lines,
        "easy_target_total": args.easy_target_total,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "questions_per_dataset": args.questions_per_dataset,
        "min_sentences": args.min_sentences,
    }
