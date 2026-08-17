"""Medium tier (Section 7.2): single-event-group explanation.

What this tier measures: given one contiguous group of log lines around a curated
anchor event, can the answer explain what that one event/condition is and what it
means -- no second group to correlate against, no cross-entity reasoning, just
reading and interpreting a single window. This is the line that separates it from
hard: medium never cites more than one evidence group.

The first tier with a model in it, and therefore the first whose records leave
generation as ``review_status=in_review``: only a human accept/edit through
``src.utils.helper_review`` can promote a record to ``verified``.

The gold answer is drafted in two stages rather than asked for directly, so where it
came from is legible rather than opaque. Stage one (``_extract_structure``) reads
the raw evidence and pulls out four fields -- event type, entity, the events
observed and the outcome -- as an intermediate, machine-checkable representation,
stored on the record as ``structured_summary``. Stage two (``_build_narrative_prompt``)
turns only that structured summary (plus the original evidence, for exact wording)
into the 1-3 sentence prose that becomes ``expected_answer``. A reviewer -- human or
model -- can now ask "does the *summary* match the evidence" and "does the
*narrative* match the summary" as two separate, smaller questions instead of one
opaque "is this paragraph right".

Every draft is checked holistically by ``src.utils.helper_validation`` against
``groundedness_model`` -- a different family from the drafting model (Section
5.5/6) -- across four dimensions (grounded, correct, relevant, sufficient), with
the question itself part of what the reviewing model sees; the per-question report
lands in ``review_dir`` where the review worksheet summarises it and
``review-apply`` refuses an ``accept`` over a dimension marked unsupported.

Evidence is a window centered on an anchor occurrence -- ``MediumTierParams.context_before``
lines before it and ``context_after`` lines after, clipped at file boundaries -- and the
drafting prompts are explicit that nothing outside that window may be used. The window
being symmetric rather than forward-only is deliberate: what led up to an event is
often exactly what "what does it mean" needs, and a forward-only window can never
show it. Each curated dataset spec can name more than one anchor
(``DatasetSpec.medium_anchors``), each standing in for a different event family, so a
dataset's medium questions are not all instances of the same single event type.
"""

import re
import sys
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec, MediumAnchorSpec
from src.params.generation_params import GenerationConfig, MediumTierParams
from src.utils import helper_validation
from src.utils.helper_evidence import evidence_ref, gold_provenance, slugify
from src.utils.helper_vllm import VllmClient

CREATED_BY = "src/generators/medium_tier.py@v1"

ORDINALS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
)

MEDIUM_DIMENSIONS = (
    (
        "grounded",
        "Is the ANSWER grounded in the EVIDENCE -- does it avoid stating "
        "anything the evidence does not show?",
    ),
    (
        "correct",
        "Does the ANSWER accurately reflect what the EVIDENCE actually shows "
        "(the right entity, the right event type, the right outcome)?",
    ),
    (
        "relevant",
        "Does the ANSWER address what the QUESTION actually asks, rather than "
        "a generic restatement of the STRUCTURED_SUMMARY that ignores the "
        "question's own wording?",
    ),
    (
        "sufficient",
        "Is the ANSWER reasonably complete -- does it avoid omitting an "
        "important part of what the evidence shows?",
    ),
)

_STRUCTURED_FIELD_PATTERN = re.compile(
    r"(?im)^\s*(EVENT_TYPE|ENTITY|OBSERVED_EVENTS|OUTCOME)\s*:\s*(.+)$"
)


def _ordinal(number: int) -> str:
    """Returns the English ordinal for a 1-based number.

    Args:
        number: 1-based ordinal number.

    Returns:
        ``"first"`` through ``"tenth"`` as words, ``"41st"``-style beyond.
    """
    if 1 <= number <= len(ORDINALS):
        return ORDINALS[number - 1]
    if 10 <= number % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


QUESTION_PHRASINGS = (
    (
        "explain",
        "In the {dataset} log, around the {ordinal} occurrence of '{anchor}': "
        "what happened and what does it mean?",
    ),
    (
        "describe",
        "Describe what is happening in the {dataset} log around the {ordinal} "
        "occurrence of '{anchor}'.",
    ),
    (
        "summarize",
        "Summarize the event shown in the {dataset} log near the {ordinal} "
        "occurrence of '{anchor}'.",
    ),
    (
        "interpret",
        "Looking at the {dataset} log around the {ordinal} occurrence of "
        "'{anchor}', what took place and what is its significance?",
    ),
    (
        "walk-through",
        "Walk me through what the {dataset} log shows around the {ordinal} "
        "occurrence of '{anchor}', and explain what it means.",
    ),
)


def _question_text(
    dataset_name: str, anchor: str, match_number: int, occurrence: int
) -> tuple[str, str]:
    """Renders one medium question's text.

    The anchor literal and the match ordinal are part of the question on
    purpose. The earlier phrasing — "Looking at these lines, what happened?" —
    was deictic: it named no searchable event, so a system evaluated black-box
    had nothing to retrieve by, and two occurrences from one dataset shared one
    question string, which downstream integrity checks rightly flag as a
    duplicate input. Naming the anchor makes the question self-contained; the
    ordinal names which real occurrence of the anchor the window starts at
    (picks are strided across the match list, so this is the match's true
    position, not the pick's index).

    ``occurrence`` cycles through ``QUESTION_PHRASINGS`` so that a dataset's
    several medium questions are not all asked in identical words — the same
    reasoning the easy tier's three phrasing families rest on, applied here
    even though ``validate.py`` does not enforce it on the semantic path.

    Args:
        dataset_name: Dataset the excerpt came from.
        anchor: The curated anchor literal.
        match_number: 1-based position of the window's anchor within every
            match of the anchor in the file.
        occurrence: 0-based index of this question among the anchor's picks,
            used to select the phrasing.

    Returns:
        Tuple ``(phrasing_family, question_text)``.
    """
    family, template = QUESTION_PHRASINGS[occurrence % len(QUESTION_PHRASINGS)]
    text = template.format(dataset=dataset_name, ordinal=_ordinal(match_number), anchor=anchor)
    return family, text


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


def _evidence_window(
    lines: list[str], anchor_idx: int, context_before: int, context_after: int
) -> list[int]:
    """Returns the indices of a symmetric window around an anchor, clipped at the file.

    Args:
        lines: The dataset's lines.
        anchor_idx: 0-based index of the anchor match.
        context_before: Lines of context to include before the anchor.
        context_after: Lines of context to include after the anchor.

    Returns:
        Window indices, ascending, including the anchor itself.
    """
    start_idx = max(0, anchor_idx - context_before)
    end_idx = min(len(lines), anchor_idx + context_after + 1)
    return list(range(start_idx, end_idx))


def _build_extraction_prompt(dataset_name: str, evidence_lines: list[str]) -> str:
    """Builds the stage-one prompt: raw evidence in, structured fields out.

    Args:
        dataset_name: Dataset the excerpt came from.
        evidence_lines: The window's lines, verbatim.

    Returns:
        The full prompt.
    """
    numbered = "\n".join(evidence_lines)
    return (
        f"You are analyzing a short excerpt of raw {dataset_name} system log lines. "
        "Read ONLY the evidence below. Extract exactly four fields about what it "
        "shows, one per line, in this exact format and nothing else:\n"
        "EVENT_TYPE: <a short label for the kind of event or condition shown>\n"
        "ENTITY: <the specific id, address, node, block, session or component the "
        "event is centered on>\n"
        "OBSERVED_EVENTS: <what actually happens in the lines, in the order it "
        "happens>\n"
        "OUTCOME: <the resulting state or condition; 'none stated' if the excerpt "
        "does not show one>\n\n"
        "If the excerpt names more than one id of the same kind, ENTITY names only "
        "the one the event is centered on -- do not merge two different ids into "
        "one entity, and do not describe something said about one id as if it "
        "happened to another.\n\n"
        f"EVIDENCE:\n{numbered}\n"
    )


def _parse_structured_summary(text: str) -> dict[str, str]:
    """Parses the stage-one completion into the four ``structured_summary`` fields.

    Matched by label rather than line position, so field order in the completion
    does not matter. A field the model never produced with a non-empty value
    falls back to ``"not stated"`` rather than an empty string, since the schema
    requires each field non-empty and an LLM occasionally drops a line.

    Args:
        text: The stage-one completion.

    Returns:
        Mapping with ``event_type``, ``entity``, ``observed_events``, ``outcome``.
    """
    found: dict[str, str] = {}
    for match in _STRUCTURED_FIELD_PATTERN.finditer(text):
        key = match.group(1).lower()
        value = match.group(2).strip()
        if key not in found and value:
            found[key] = value
    return {
        "event_type": found.get("event_type", "not stated"),
        "entity": found.get("entity", "not stated"),
        "observed_events": found.get("observed_events", "not stated"),
        "outcome": found.get("outcome", "not stated"),
    }


def _extract_structure(
    client: VllmClient, dataset_name: str, evidence_lines: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Runs stage one: drafts and parses the structured summary.

    Args:
        client: vLLM client; drafts always run on ``gold_draft_model``.
        dataset_name: Dataset the excerpt came from.
        evidence_lines: The window's lines, verbatim.

    Returns:
        Tuple ``(structured_summary, model)``.
    """
    draft = client.draft(_build_extraction_prompt(dataset_name, evidence_lines))
    return _parse_structured_summary(draft["text"]), draft["model"]


def _build_narrative_prompt(
    dataset_name: str, structured: dict[str, str], evidence_lines: list[str]
) -> str:
    """Builds the stage-two prompt: structured summary in, prose out.

    Args:
        dataset_name: Dataset the excerpt came from.
        structured: Output of ``_parse_structured_summary``.
        evidence_lines: The window's lines, verbatim, shown again for exact wording.

    Returns:
        The full prompt.
    """
    numbered = "\n".join(evidence_lines)
    summary = (
        f"EVENT_TYPE: {structured['event_type']}\n"
        f"ENTITY: {structured['entity']}\n"
        f"OBSERVED_EVENTS: {structured['observed_events']}\n"
        f"OUTCOME: {structured['outcome']}"
    )
    return (
        "Below is a structured summary already extracted from a short excerpt of "
        f"raw {dataset_name} system log lines, plus the original evidence for exact "
        "wording. Using ONLY the summary and the evidence -- introduce no fact "
        "that is not in either -- write 1 to 3 sentences of plain prose explaining "
        "what happened and what it means.\n\n"
        f"STRUCTURED SUMMARY:\n{summary}\n\nEVIDENCE:\n{numbered}\n\n"
        "Answer with only the explanation (1-3 sentences), no preamble, no bullet "
        "points."
    )


def _pick_anchor_occurrences(
    match_indices: list[int], params: MediumTierParams
) -> list[tuple[int, int]]:
    """Spreads the requested number of picks across the matches.

    Consecutive matches are usually one burst of the same event, so drafting from
    the first N of them would produce N near-identical questions over overlapping
    windows. Striding across the match list instead gives each question its own
    region of the file. Each pick carries its 1-based position within the match
    list because the question text names that ordinal, and the pick's index would
    be the wrong number: pick #2 under a stride of 40 is the 41st match.

    Args:
        match_indices: Every anchor match, ascending.
        params: Medium-tier knobs; ``questions_per_dataset`` is the target count.

    Returns:
        Up to ``questions_per_dataset`` pairs of ``(match_number, line_index)``.
    """
    step = max(1, len(match_indices) // params.questions_per_dataset)
    picks = []
    for i in range(0, len(match_indices), step):
        picks.append((i + 1, match_indices[i]))
        if len(picks) >= params.questions_per_dataset:
            break
    return picks


def _build_one_anchor_records(
    view: CorpusView,
    anchor: MediumAnchorSpec,
    config: GenerationConfig,
    client: VllmClient,
) -> list[dict[str, Any]]:
    """Builds every medium record for one curated anchor.

    Args:
        view: The dataset's corpus.
        anchor: One curated anchor spec.
        config: Generation config; its ``medium`` field carries the tier knobs.
        client: vLLM client used for both draft stages and the quality check.

    Returns:
        The drafted records for this anchor, all ``review_status=in_review``.
    """
    params = config.medium
    match_indices = _find_matches(view.lines, anchor.literal)
    if not match_indices:
        print(
            f"  [prune] {view.name}: medium anchor '{anchor.literal}' has 0 matches, "
            f"skipping",
            file=sys.stderr,
        )
        return []

    slug = slugify(anchor.literal)
    records = []
    for occurrence, (match_number, anchor_idx) in enumerate(
        _pick_anchor_occurrences(match_indices, params)
    ):
        window_indices = _evidence_window(
            view.lines, anchor_idx, params.context_before, params.context_after
        )
        evidence_lines = [view.lines[i] for i in window_indices]
        evidence_text = "\n".join(evidence_lines)
        group_id = f"{view.key}:semantic:{slug}_{occurrence}"
        refs = [
            evidence_ref(view.key, i + 1, view.lines[i], group_id)
            for i in window_indices
        ]

        structured, extraction_model = _extract_structure(
            client, view.name, evidence_lines
        )
        narrative = client.draft(
            _build_narrative_prompt(view.name, structured, evidence_lines)
        )

        question_id = f"{view.key}_v1_semantic_{slug}_{occurrence}"
        phrasing_family, question_text = _question_text(
            view.name, anchor.literal, match_number, occurrence
        )

        context = {
            "QUESTION": question_text,
            "STRUCTURED_SUMMARY": (
                f"EVENT_TYPE: {structured['event_type']}\n"
                f"ENTITY: {structured['entity']}\n"
                f"OBSERVED_EVENTS: {structured['observed_events']}\n"
                f"OUTCOME: {structured['outcome']}"
            ),
            "EVIDENCE": evidence_text,
            "ANSWER": narrative["text"],
        }
        validation = helper_validation.run_checks(client, context, MEDIUM_DIMENSIONS)
        helper_validation.write_report(
            config.review_dir, question_id, [group_id], narrative["model"], validation
        )

        records.append(
            {
                "id": question_id,
                "question": question_text,
                "routing_path": "semantic",
                "answer_type": "explanation",
                "task": "Summarization",
                "difficulty": "medium",
                "phrasing_family": phrasing_family,
                "review_status": "in_review",
                "reviewers": [config.reviewer],
                "expected_answer": narrative["text"],
                "gold_provenance": gold_provenance(
                    method="independent_model_then_human",
                    created_by=CREATED_BY,
                    created_at=config.created_at,
                    corpus_sha256=view.sha256,
                    model=narrative["model"],
                ),
                "structured_summary": structured,
                "evidence": {"refs": refs},
                "validation": validation,
            }
        )
        print(
            f"  [{view.name}] drafted semantic question '{anchor.literal}' "
            f"#{occurrence} ({len(evidence_lines)} evidence lines, "
            f"extraction_model={extraction_model['name']})"
        )
    return records


def build_medium_records(
    view: CorpusView,
    spec: DatasetSpec,
    config: GenerationConfig,
    client: VllmClient,
) -> list[dict[str, Any]]:
    """Builds every medium-tier record for one dataset, across every curated anchor.

    A dataset with no ``medium_anchors`` contributes nothing and says so — the
    tier is anchored on curated events, and inventing a fallback anchor would
    mean drafting an explanation of whatever happened to be at the top of the
    file.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        config: Generation config; its ``medium`` field carries the tier knobs.
        client: vLLM client used for the gold drafts and the quality check.

    Returns:
        The drafted records, all ``review_status=in_review``.
    """
    records: list[dict[str, Any]] = []
    for anchor in spec.medium_anchors:
        records.extend(_build_one_anchor_records(view, anchor, config, client))
    return records
