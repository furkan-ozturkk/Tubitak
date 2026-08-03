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

Each group's evidence block is exactly the cited lines plus a header naming the
group and how many sample lines follow — nothing else. An earlier revision put the
group's *total* match count, first line and last line in that header, and the
drafted "syntheses" duly answered by reading the header: "80 lines versus 23, so
the first source is more anomalous" is arithmetic over a number the evidence does
not show and a black-box system under evaluation could not retrieve. The gold has
to be derivable from the cited lines alone, so the header now states nothing the
lines do not, the prompt forbids claims about unseen totals, and the question
templates ask about the patterns in the shown samples rather than about absolute
volume.

The drafted answer goes through the shared claim-by-claim groundedness check
(``src.utils.helper_groundedness``): every sentence is judged independently by
``groundedness_model``, a different family from the drafting model (Section
5.5/6), and the per-question report's verdict is always ``needs_human_review``.
"""

import re
import sys
from collections import defaultdict
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec, HardGroupSpec
from src.utils.helper_groundedness import check_claims, write_report
from src.params.generation_params import GenerationConfig
from src.utils.helper_evidence import evidence_ref, gold_provenance, slugify
from src.utils.helper_ollama import OllamaClient

CREATED_BY = "src/generators/hard_tier.py@v1"

HARD_PHRASING_FAMILIES = (
    "compare",
    "how-differ",
    "contrast",
    "both-appear",
    "examine-versus",
)


def _select_group_sets(
    lines: list[str], hard_spec: HardGroupSpec, max_sets: int
) -> list[list[tuple[str, list[int]]]]:
    """Groups lines by the spec's key regex and picks non-overlapping group sets.

    Qualifying groups are ranked largest-first (key name breaking ties) and then
    chunked into consecutive, non-overlapping sets of ``hard_spec.num_groups``
    entities each: the first set gets the two largest groups, the second set the
    next two, and so on. Chunking rather than resampling means no entity is ever
    compared against itself twice, and the ranking means the richest-evidence
    comparisons are always drafted first when ``max_sets`` caps the total below
    what the corpus could support.

    Args:
        lines: The dataset's lines.
        hard_spec: The hard-question spec, carrying the key regex, the minimum
            lines per group and how many groups one question needs.
        max_sets: Upper bound on how many sets to return.

    Returns:
        Up to ``max_sets`` lists of ``(key, line_indices)`` pairs, each list of
        length ``hard_spec.num_groups``; empty when the corpus does not contain
        enough qualifying groups to build even one question.
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
    ranked = sorted(qualifying.items(), key=lambda item: (-len(item[1]), item[0]))

    sets = []
    n = hard_spec.num_groups
    for start in range(0, len(ranked), n):
        chunk = ranked[start : start + n]
        if len(chunk) < n:
            break
        sets.append(chunk)
        if len(sets) >= max_sets:
            break
    return sets


def _build_evidence_block(
    view: CorpusView,
    key: str,
    indices: list[int],
    hard_spec: HardGroupSpec,
    group_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Builds one group's evidence refs and the text block shown to the model.

    The block opens with a header naming the group and the number of sample lines
    that follow, then the cited lines — and nothing more. The group's total match
    count is deliberately absent: stating it hands the model a comparison the
    evidence itself does not support, and the gold answer must rest only on lines
    the record cites (see the module docstring). Refs and shown lines are cut to
    the same set, so the prompt cannot contain a line the record does not cite.

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
    summary = f"[Group: {key}] {len(cited)} sample lines:"
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
        "groups. Each group is a header naming the group followed by its sample lines. "
        "Read ONLY the evidence below. Do not speculate about anything not shown, and do "
        "not claim totals, counts or frequencies beyond the sample lines you can see — "
        "compare the groups by the timing, actors and event patterns visible in those "
        f"lines. Write an answer of at least {min_sentences} sentences that correlates "
        "the groups, explicitly uses facts from every group, and proposes a root-cause "
        "hypothesis.\n\n"
        f"QUESTION: {question_text}\n\nEVIDENCE:\n{evidence_text}\n\nAnswer:"
    )


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
        sets = _select_group_sets(view.lines, hard_spec, params.pairs_per_dataset)
        if not sets:
            print(
                f"  [quota_unmet] {view.name}/{hard_spec.spec_id}: fewer than "
                f"{hard_spec.num_groups} groups with >={hard_spec.min_lines_per_group} lines",
                file=sys.stderr,
            )
            continue

        for occurrence, selected in enumerate(sets):
            keys = [key for key, _ in selected]
            group_ids = [
                f"{view.key}:hard:{hard_spec.spec_id}:{occurrence}:{slugify(key)}"
                for key in keys
            ]
            all_refs: list[dict[str, Any]] = []
            evidence_blocks = []
            for (key, indices), group_id in zip(selected, group_ids):
                refs, block = _build_evidence_block(
                    view, key, indices, hard_spec, group_id
                )
                all_refs.extend(refs)
                evidence_blocks.append(block)
            evidence_text = "\n\n".join(evidence_blocks)

            templates = hard_spec.question_templates
            phrasing_index = occurrence % len(templates)
            question_text = templates[phrasing_index].format(
                **{f"key{index}": key for index, key in enumerate(keys)}
            )
            phrasing_family = HARD_PHRASING_FAMILIES[
                phrasing_index % len(HARD_PHRASING_FAMILIES)
            ]

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

            question_id = f"{view.key}_v1_hard_{hard_spec.spec_id}_{occurrence}"
            claims, reviewer_model = check_claims(
                client, answer_text, evidence_text, group_ids
            )
            write_report(
                config.review_dir,
                question_id,
                group_ids,
                draft["model"],
                reviewer_model,
                claims,
            )

            records.append(
                {
                    "id": question_id,
                    "question": question_text,
                    "routing_path": "semantic",
                    "answer_type": "synthesis",
                    "task": hard_spec.task,
                    "difficulty": "hard",
                    "phrasing_family": phrasing_family,
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
                f"  [{view.name}] drafted hard question '{hard_spec.spec_id}' #{occurrence} "
                f"({len(claims)} claims, groups={keys})"
            )
    return records
