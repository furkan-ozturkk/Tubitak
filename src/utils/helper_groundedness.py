"""Claim-by-claim groundedness checking shared by the model-drafted tiers.

A client-side helper beside ``helper_ollama`` and ``helper_review`` rather than a
generator: it produces no records, it orchestrates the reviewing model over a
draft and files the report the review flow reads.

Both the medium and the hard tier draft their gold with ``gold_draft_model`` and
must have every sentence of that draft checked by ``groundedness_model`` — a
different family, verified against server metadata (Section 5.5/6). The check
itself is tier-agnostic: split the draft into sentences, ask the reviewing model
whether the evidence supports each one, and write a per-question report whose
verdict is always ``needs_human_review``. A second model's agreement is evidence
for the human reviewer, never a substitute for them; ``helper_review`` surfaces
the report in the worksheet and refuses an ``accept`` while any claim stands
unsupported.

A claim is linked to the group ids it could have drawn on rather than to one
line: the check returns a verdict against the whole evidence block and does not
resolve which line supports a claim, so naming one would assert a precision it
did not establish. An unsupported claim carries no groups at all.
"""

from pathlib import Path
from typing import Any

from src.utils.helper_evidence import split_sentences
from src.utils.helper_ollama import OllamaClient
from src.utils.helper_run import write_json


def check_claims(
    client: OllamaClient,
    answer_text: str,
    evidence_text: str,
    group_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Checks each sentence of a drafted answer against its evidence.

    Args:
        client: Ollama client; the check always runs on ``groundedness_model``.
        answer_text: The drafted gold answer.
        evidence_text: The evidence the answer had to be derived from.
        group_ids: The question's evidence group ids.

    Returns:
        Tuple ``(claims, reviewer_model)``, the second being the reviewing
        model's provenance block, or ``None`` when the answer held no claims.
    """
    claims: list[dict[str, Any]] = []
    reviewer_model: dict[str, Any] | None = None
    for claim in split_sentences(answer_text):
        verdict, result = client.groundedness_check(claim, evidence_text)
        reviewer_model = result["model"]
        claims.append(
            {
                "text": claim,
                "supported": verdict,
                "candidate_group_ids": [] if verdict == "no" else list(group_ids),
            }
        )
    return claims, reviewer_model


def write_report(
    review_dir: Path,
    question_id: str,
    group_ids: list[str],
    draft_model: dict[str, Any],
    reviewer_model: dict[str, Any] | None,
    claims: list[dict[str, Any]],
) -> None:
    """Writes one question's groundedness report where the review flow reads it.

    Both models' provenance is recorded: a report that names only the drafting
    model documents half of the integrity claim.

    Args:
        review_dir: Directory of per-question reports
            (``GenerationConfig.review_dir``).
        question_id: The record id, which is also the report's filename.
        group_ids: The question's evidence group ids.
        draft_model: The drafting model's provenance block.
        reviewer_model: The reviewing model's provenance block, or ``None``.
        claims: Output of ``check_claims``.
    """
    write_json(
        Path(review_dir) / f"{question_id}.json",
        {
            "question_id": question_id,
            "group_ids": group_ids,
            "draft_model": draft_model,
            "reviewer_model": reviewer_model,
            "claims": claims,
            "verdict": "needs_human_review",
        },
    )
