"""Hard tier (Section 7.3): multi-event / root-cause / synthesis.

The evidence groups are selected before the question is drafted, not after. A hard
question has to correlate at least two distinct event groups — the schema and
``validate.py`` both enforce that — so the groups are what the question is built
around; drafting first and hunting for supporting lines afterwards is how a
"synthesis" question ends up resting on one group.

Group selection is deterministic. Lines are keyed by the spec's regex, groups below
the minimum line count are discarded, and the survivors are ranked by size with the
key name breaking ties. Nothing is sampled, so the same corpus yields the same
groups on every run (Section 2).

Each group's evidence carries a deterministic feature summary — total lines, the
cited window, and the first and last cited line — because the questions ask about
volume, timing and an observation window while the prompt shows only the first few
lines of each group. Handing the model four lines per side and asking which source
is more anomalous invites an answer about a volume it cannot see; the summary states
the counts explicitly so the comparison rests on something recorded.

The drafted answer is split into sentences and each one is checked independently by
``groundedness_model``, a different family from the drafting model (Section 5.5/6).
The report is written per question and its verdict is always
``needs_human_review``: a second model's agreement is evidence for the reviewer,
never a substitute for them. Both models' provenance is recorded, and a claim is
linked to the group ids it could have come from rather than to one arbitrary line —
the check does not resolve which line supports a claim, and naming one would assert
a precision it did not establish.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec, HardGroupSpec
from src.params.generation_params import GenerationConfig
from src.utils.helper_evidence import (
    evidence_ref,
    gold_provenance,
    slugify,
    split_sentences,
)
from src.utils.helper_ollama import OllamaClient
from src.utils.helper_run import write_json

CREATED_BY = "src/generators/hard_tier.py@v1"


def _select_groups(
    lines: list[str], hard_spec: HardGroupSpec
) -> list[tuple[str, list[int]]] | None:
    """Groups lines by the spec's key regex and picks the largest qualifying groups.

    Args:
        lines: The dataset's lines.
        hard_spec: The hard-question spec, carrying the key regex, the minimum lines
            per group and how many groups the question needs.

    Returns:
        ``(key, line_indices)`` pairs, largest first with the key name breaking ties,
        or ``None`` when the corpus does not contain enough qualifying groups to
        build the question at all.
    """
    pattern = re.compile(hard_spec.extract_key_regex)
    by_key = defaultdict(list)
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            by_key[match.group("key")].append(index)

    qualifying = {
        key: indices
        for key, indices in by_key.items()
        if len(indices) >= hard_spec.min_lines_per_group
    }
    if len(qualifying) < hard_spec.num_groups:
        return None

    ranked = sorted(qualifying.items(), key=lambda item: (-len(item[1]), item[0]))
    return ranked[: hard_spec.num_groups]


def _build_evidence_block(
    view: CorpusView,
    key: str,
    indices: list[int],
    hard_spec: HardGroupSpec,
    group_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Builds one group's evidence refs and the text block shown to the model.

    The block opens with the group's deterministic feature summary and then the
    cited lines. Refs and shown lines are cut to the same set, so the prompt cannot
    contain a line the record does not cite.

    Args:
        view: The dataset's corpus.
        key: The group key (a block id, container id, source address, ...).
        indices: Every line index in the group.
        hard_spec: Spec carrying ``evidence_lines_per_group``.
        group_id: The group's evidence identifier.

    Returns:
        Tuple ``(refs, text_block)``.
    """
    cited = indices[: hard_spec.evidence_lines_per_group]
    refs = [evidence_ref(view.key, i + 1, view.lines[i], group_id) for i in cited]
    summary = (
        f"[Group: {key}] total_matching_lines={len(indices)} "
        f"shown={len(cited)} "
        f"first_line={indices[0] + 1} last_line={indices[-1] + 1}"
    )
    body = "\n".join(view.lines[i] for i in cited)
    return refs, f"{summary}\n{body}"


def _build_prompt(
    dataset_name: str,
    group_count: int,
    question_text: str,
    evidence_text: str,
    min_sentences: int,
) -> str:
    """Builds the drafting prompt for one hard question.

    Args:
        dataset_name: Dataset the groups came from.
        group_count: How many event groups the answer must use.
        question_text: The rendered question.
        evidence_text: Every group's summary and evidence block, concatenated.
        min_sentences: Minimum sentences demanded of the answer.

    Returns:
        The full prompt.
    """
    return (
        f"You are analyzing raw {dataset_name} log lines from {group_count} related event "
        "groups. Each group starts with a header stating how many lines it has in total and "
        "which of them are shown. Read ONLY the evidence below; do not speculate about "
        "anything not shown, and when comparing volume use the stated totals rather than the "
        f"number of lines displayed. Write an answer of at least {min_sentences} sentences "
        "that correlates the groups, explicitly uses facts from every group, and proposes a "
        "root-cause hypothesis.\n\n"
        f"QUESTION: {question_text}\n\nEVIDENCE:\n{evidence_text}\n\nAnswer:"
    )


def _run_groundedness_check(
    client: OllamaClient,
    answer_text: str,
    evidence_text: str,
    group_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Checks each sentence of the drafted answer against the evidence.

    ``candidate_group_ids`` holds the groups a claim could have drawn on, not the
    line that supports it. The check returns a verdict on the whole evidence block
    and does not identify a supporting line, so an unsupported claim carries no
    groups at all and a supported one carries every group it was shown.

    Args:
        client: Ollama client; the check always runs on ``groundedness_model``.
        answer_text: The drafted gold answer.
        evidence_text: The evidence the answer had to be derived from.
        group_ids: The question's evidence group ids.

    Returns:
        Tuple ``(claims, reviewer_model)``, the second being the reviewing model's
        provenance block, or ``None`` when the answer held no claims.
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


def build_hard_records(
    view: CorpusView,
    spec: DatasetSpec,
    config: GenerationConfig,
    client: OllamaClient,
) -> list[dict[str, Any]]:
    """Builds every hard-tier record for one dataset.

    A spec whose groups cannot be filled is reported as ``quota_unmet`` and skipped
    rather than relaxed: lowering ``min_lines_per_group`` to make a question fit
    would produce a correlation question over two lines.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        config: Generation config; its ``hard`` field carries the tier knobs and its
            ``review_dir`` is where groundedness reports are written.
        client: Ollama client used for the draft and the groundedness check.

    Returns:
        The drafted records, all ``review_status=in_review``.
    """
    params = config.hard
    if not spec.hard_groups:
        return []

    records = []
    for hard_spec in spec.hard_groups:
        selected = _select_groups(view.lines, hard_spec)
        if selected is None:
            print(
                f"  [quota_unmet] {view.name}/{hard_spec.spec_id}: fewer than "
                f"{hard_spec.num_groups} groups with >={hard_spec.min_lines_per_group} lines",
                file=sys.stderr,
            )
            continue

        keys = [key for key, _ in selected]
        group_ids = [
            f"{view.key}:hard:{hard_spec.spec_id}:{slugify(key)}" for key in keys
        ]
        all_refs: list[dict[str, Any]] = []
        evidence_blocks = []
        for (key, indices), group_id in zip(selected, group_ids):
            refs, block = _build_evidence_block(view, key, indices, hard_spec, group_id)
            all_refs.extend(refs)
            evidence_blocks.append(block)
        evidence_text = "\n\n".join(evidence_blocks)

        question_text = hard_spec.question_template.format(
            **{f"key{index}": key for index, key in enumerate(keys)}
        )

        draft = client.draft(
            _build_prompt(
                view.name,
                len(keys),
                question_text,
                evidence_text,
                params.min_sentences,
            )
        )
        answer_text = draft["text"]

        question_id = f"{view.key}_v1_hard_{hard_spec.spec_id}"
        claims, reviewer_model = _run_groundedness_check(
            client, answer_text, evidence_text, group_ids
        )
        write_json(
            Path(config.review_dir) / f"{question_id}.json",
            {
                "question_id": question_id,
                "group_ids": group_ids,
                "draft_model": draft["model"],
                "reviewer_model": reviewer_model,
                "claims": claims,
                "verdict": "needs_human_review",
            },
        )

        records.append(
            {
                "id": question_id,
                "question": question_text,
                "routing_path": "semantic",
                "answer_type": "synthesis",
                "task": hard_spec.task,
                "difficulty": "hard",
                "phrasing_family": "hard-synthesis",
                "review_status": "in_review",
                "reviewers": [config.reviewer],
                "expected_answer": answer_text,
                "gold_provenance": gold_provenance(
                    method="independent_model_then_human",
                    created_by=CREATED_BY,
                    created_at=config.created_at,
                    corpus_sha256=view.sha256,
                    model=draft["model"],
                ),
                "evidence": {"refs": all_refs},
            }
        )
        print(
            f"  [{view.name}] drafted hard question '{hard_spec.spec_id}' "
            f"({len(claims)} claims, groups={keys})"
        )
    return records
