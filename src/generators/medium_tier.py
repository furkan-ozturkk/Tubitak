"""Medium tier (Section 7.2): single-event-family explanation / summary.

The first tier with a model in it, and therefore the first whose records leave
generation as ``review_status=in_review``: the gold answer is a draft by
``gold_draft_model``, and only a human accept/edit through
``src.utils.helper_review`` can promote it to ``verified``.

Evidence is a contiguous window starting at an anchor occurrence, capped at
``MediumTierParams.window_size``, and the drafting prompt is explicit that
nothing outside that window may be used. The cap is not a token-budget
convenience: a record's ``evidence.refs`` is the claim about what its answer was
derived from, so anything the model saw has to be in there.
"""

import sys
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec
from src.params.generation_params import GenerationConfig, MediumTierParams
from src.utils.helper_evidence import evidence_ref, gold_provenance, slugify
from src.utils.helper_ollama import OllamaClient

CREATED_BY = "src/generators/medium_tier.py@v1"


def _find_matches(lines: list[str], literal: str) -> list[int]:
    """Returns the 0-based indices of every line containing ``literal``.

    Case-insensitive, and reading the file rather than Postgres: the medium tier
    needs contiguous windows around a match, which is a positional question about
    the file, not an aggregate the database is the authority on.

    Args:
        lines: The dataset's lines.
        literal: Anchor literal to search for.

    Returns:
        Matching indices, ascending.
    """
    needle = literal.lower()
    return [i for i, line in enumerate(lines) if needle in line.lower()]


def _evidence_window(lines: list[str], start_idx: int, size: int) -> list[int]:
    """Returns the indices of a window of lines, clipped at end of file.

    Args:
        lines: The dataset's lines.
        start_idx: 0-based index of the anchor match.
        size: Maximum window length.

    Returns:
        Window indices, ascending.
    """
    end_idx = min(len(lines), start_idx + size)
    return list(range(start_idx, end_idx))


def _build_prompt(dataset_name: str, evidence_lines: list[str]) -> str:
    """Builds the drafting prompt for one medium question.

    Args:
        dataset_name: Dataset the excerpt came from, named so the model does not
            have to infer the system from the log format.
        evidence_lines: The window's lines, verbatim.

    Returns:
        The full prompt.
    """
    numbered = "\n".join(evidence_lines)
    return (
        f"You are analyzing a short excerpt of raw {dataset_name} system log lines. "
        "Read ONLY the evidence below and answer using nothing but what these lines show "
        "-- do not speculate about anything not directly stated. In 1 to 3 sentences, "
        "explain what event or condition this excerpt shows and what it means.\n\n"
        f"EVIDENCE:\n{numbered}\n\n"
        "Answer with only the explanation (1-3 sentences), no preamble, no bullet points."
    )


def _pick_anchor_occurrences(match_indices: list[int], params: MediumTierParams) -> list[int]:
    """Spreads the requested number of picks across the matches.

    Consecutive matches are usually one burst of the same event, so drafting from
    the first N of them would produce N near-identical questions over overlapping
    windows. Striding across the match list instead gives each question its own
    region of the file.

    Args:
        match_indices: Every anchor match, ascending.
        params: Medium-tier knobs; ``questions_per_dataset`` is the target count.

    Returns:
        Up to ``questions_per_dataset`` anchor indices.
    """
    step = max(1, len(match_indices) // params.questions_per_dataset)
    picks = []
    for i in range(0, len(match_indices), step):
        picks.append(match_indices[i])
        if len(picks) >= params.questions_per_dataset:
            break
    return picks


def build_medium_records(
    view: CorpusView,
    spec: DatasetSpec,
    config: GenerationConfig,
    client: OllamaClient,
) -> list[dict[str, Any]]:
    """Builds every medium-tier record for one dataset.

    A dataset with no ``medium_anchor_literal``, or whose anchor matches nothing,
    contributes nothing and says so — the tier is anchored on curated events, and
    inventing a fallback anchor would mean drafting an explanation of whatever
    happened to be at the top of the file.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        config: Generation config; its ``medium`` field carries the tier knobs.
        client: Ollama client used for the gold draft.

    Returns:
        The drafted records, all ``review_status=in_review``.
    """
    params = config.medium
    if not spec.medium_anchor_literal:
        return []

    match_indices = _find_matches(view.lines, spec.medium_anchor_literal)
    if not match_indices:
        print(
            f"  [prune] {view.name}: medium anchor '{spec.medium_anchor_literal}' has 0 matches, "
            f"skipping",
            file=sys.stderr,
        )
        return []

    slug = slugify(spec.medium_anchor_literal)
    records = []
    for occurrence, start_idx in enumerate(_pick_anchor_occurrences(match_indices, params)):
        window_indices = _evidence_window(view.lines, start_idx, params.window_size)
        evidence_lines = [view.lines[i] for i in window_indices]
        group_id = f"{view.key}:semantic:{slug}_{occurrence}"
        refs = [
            evidence_ref(view.key, i + 1, view.lines[i], group_id) for i in window_indices
        ]

        draft = client.draft(_build_prompt(view.name, evidence_lines))

        records.append(
            {
                "id": f"{view.key}_v1_semantic_{slug}_{occurrence}",
                "question": (
                    f"Looking at these {view.name} log lines, what happened and what does it mean?"
                ),
                "routing_path": "semantic",
                "answer_type": "explanation",
                "task": "Summarization",
                "difficulty": "medium",
                "phrasing_family": "single-event-summary",
                "review_status": "in_review",
                "reviewers": [config.reviewer],
                "expected_answer": draft["text"],
                "gold_provenance": gold_provenance(
                    method="independent_model_then_human",
                    created_by=CREATED_BY,
                    created_at=config.created_at,
                    corpus_sha256=view.sha256,
                    model=draft["model"],
                ),
                "evidence": {"refs": refs},
            }
        )
        print(
            f"  [{view.name}] drafted semantic question {occurrence} "
            f"({len(evidence_lines)} evidence lines)"
        )
    return records
