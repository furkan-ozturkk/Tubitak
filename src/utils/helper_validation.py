"""Post-hoc quality checking shared by all three question tiers.

A client-side helper beside ``helper_vllm`` and ``helper_review`` rather than a
generator: it produces no records, it orchestrates the reviewing model over an
already-built question/answer/evidence triple and files the report the review
flow reads.

Every tier -- easy included, now that ``easy_tier.py`` calls this too -- runs its
finished record through ``VllmClient.check_dimensions`` on ``groundedness_model``, a
different family from whatever produced the answer (Section 5.5/6; for easy, the
answer itself came from SQL, not a model, so this check is a safety net rather than
a precondition for correctness). Each tier supplies its own list of quality
dimensions and its own context blocks; what is shared is the mechanism: one call
evaluates every dimension at once against the full context (question, evidence,
answer, and for medium the structured summary), and the per-question report's
outcome is a set of ``{"dimension", "verdict"}`` entries, never a single collapsed
pass/fail.

This module replaces the earlier per-sentence claim check (formerly
``helper_groundedness.check_claims``): that mode showed the reviewer one claim
sentence and the evidence, but never the question the answer was supposed to
address, so a claim could read as locally true while the check stayed blind to an
answer that did not actually answer what was asked. Holistic, question-aware
dimensions close that gap, at the cost of losing the earlier mode's per-sentence
citation -- what is unsupported is now a whole dimension of the answer, not a single
sentence, which is the right resolution for a check whose job is "does the record
hold together end to end", not "footnote every clause".

A model's verdict is embedded directly on the record (``record["validation"]``,
Section 6) so it travels with the record as the single source of truth; this
module additionally writes the same result to ``review_dir`` as a per-question
report, kept as fuller-provenance audit detail beyond what the record itself
carries.
"""

from pathlib import Path
from typing import Any

from src.utils.helper_run import write_json
from src.utils.helper_vllm import VllmClient


def run_checks(
    client: VllmClient,
    context: dict[str, str],
    dimensions: list[tuple[str, str]],
) -> dict[str, Any]:
    """Runs one tier's quality dimensions and returns the record's ``validation`` block.

    Args:
        client: vLLM client; the check always runs on ``groundedness_model``.
        context: Named context blocks shown to the model (e.g. ``QUESTION``,
            ``EVIDENCE``, ``ANSWER``, ``STRUCTURED_SUMMARY``).
        dimensions: ``(key, question)`` pairs, one per quality dimension this
            tier checks.

    Returns:
        A ``{"checks": [...], "model": {...}}`` mapping, exactly the shape
        ``config/question_schema.json`` requires of ``record["validation"]``.
    """
    checks, reviewer_model = client.check_dimensions(context, dimensions)
    return {"checks": checks, "model": reviewer_model}


def has_unsupported_check(validation: dict[str, Any] | None) -> bool:
    """Reports whether any dimension of a ``validation`` block came back ``"no"``.

    Args:
        validation: A record's ``validation`` block, or ``None``.

    Returns:
        ``True`` if at least one check's verdict is ``"no"``.
    """
    if not validation:
        return False
    return any(check.get("verdict") == "no" for check in validation.get("checks", []))


def write_report(
    review_dir: Path,
    question_id: str,
    group_ids: list[str],
    subject_model: dict[str, Any] | None,
    validation: dict[str, Any],
) -> None:
    """Writes one question's validation report where the review flow reads it.

    Both models' provenance is recorded where both exist: a report that names only
    the reviewing model documents half of the integrity claim for medium/hard,
    whose answer is itself model-drafted. Easy has no ``subject_model`` (its answer
    is SQL-computed), so that field is simply omitted rather than faked.

    Args:
        review_dir: Directory of per-question reports
            (``GenerationConfig.review_dir``).
        question_id: The record id, which is also the report's filename.
        group_ids: The question's evidence group ids.
        subject_model: The provenance block of whatever produced the answer being
            checked (the gold-draft model for medium/hard), or ``None`` for easy.
        validation: Output of ``run_checks`` -- the record's own ``validation``
            block, written here verbatim plus the surrounding identifiers.
    """
    report: dict[str, Any] = {
        "question_id": question_id,
        "group_ids": group_ids,
        "reviewer_model": validation["model"],
        "checks": validation["checks"],
    }
    if subject_model is not None:
        report["subject_model"] = subject_model
    write_json(Path(review_dir) / f"{question_id}.json", report)
