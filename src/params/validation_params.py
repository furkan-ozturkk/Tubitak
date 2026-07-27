"""Configuration parameters for the validate command (Sections 2/6)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.helper_splits import TEST_FRACTION


@dataclass(frozen=True)
class ValidationConfig:
    """What is validated, against what, and how strictly.

    ``corpus_dir`` decides whether the corpus-integrity and answer layers can run at
    all. It is not optional in practice — the default points at the mounted corpus —
    but the field exists so the validator can be pointed elsewhere, and a run that
    lands somewhere without ``*.log`` files fails rather than reporting a schema-only
    pass.

    ``test_fraction`` must match the value generation used. It is what the split layer
    recomputes the dev/test boundary with, and a validator running a different
    fraction would report every record's split as wrong.

    Attributes:
        questions: File(s) or glob pattern(s) to validate.
        schema: ``question_schema.json`` path.
        corpus_dir: Corpus directory for the file-level and answer-level checks.
        manifest: ``corpus_manifest.json`` path. When set, every dataset it pins must
            be cited by the validated questions.
        strict: Promote the Section 2/7.4 phrasing-diversity finding to an error.
        report: JSON report path.
        test_fraction: Fraction of evidence-group components assigned to test.
    """

    questions: tuple[str, ...]
    schema: Path
    corpus_dir: Path | None
    manifest: Path | None = None
    strict: bool = False
    report: Path | None = None
    test_fraction: float = TEST_FRACTION


def get_validation_params(args: Any) -> ValidationConfig:
    """Constructs a ValidationConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace. ``questions`` has already been resolved to
            ``[--dataset]`` by ``config.args._resolve_paths`` when the flag was
            omitted, so this never has to guess what "validate the dataset" means.

    Returns:
        ValidationConfig populated from args.
    """
    return ValidationConfig(
        questions=tuple(args.questions),
        schema=args.schema,
        corpus_dir=args.corpus_dir,
        manifest=args.manifest,
        strict=args.strict,
        report=args.report,
        test_fraction=(
            ValidationConfig.test_fraction
            if args.test_fraction is None
            else args.test_fraction
        ),
    )
