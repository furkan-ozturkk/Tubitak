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
from src.utils.helper_vllm import VllmClient

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
    per_side: int,
    group_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Builds one group's evidence refs and the text block shown to the model.

    Cites the group's first and last ``per_side`` lines (all of it, unsplit,
    when the group is small enough that the two halves would overlap). A
    fixed-size cut of only a group's first few lines understated a burst that
    actually ran for minutes: the record's evidence has to carry the group's
    true start and end, not just its opening. Citing literally every line of
    a group with hundreds of matches was tried next — with ``num_ctx`` raised
    so the prompt would not be silently truncated, the request succeeded, but
    a wall of hundreds of near-identical lines measurably degraded the draft:
    it lost track of which lines belonged to which group and answered a
    different question than the one asked. First-N-plus-last-N is the
    version that actually drafts a correct answer: the group's true start and
    end are always in evidence via the timestamps on the boundary lines, and
    needs no explicit gap marker or line count for the model to read — stating
    the omitted count would hand it a number to compare group sizes by
    instead of reading what the lines describe (see the module docstring and
    the prompt below). Refs are cut to the same lines the text shows, so the
    prompt can never contain a line the record does not cite.

    Args:
        view: The dataset's corpus.
        key: The group key (a block id, container id, source address, ...).
        indices: Every line index in the group.
        per_side: How many of the group's first and last lines to cite.
        group_id: The group's evidence identifier.

    Returns:
        Tuple ``(refs, text_block)``.
    """
    if len(indices) <= 2 * per_side:
        cited = indices
    else:
        cited = indices[:per_side] + indices[-per_side:]
    refs = [evidence_ref(view.key, i + 1, view.lines[i], group_id) for i in cited]
    summary = f"[Group: {key}] {len(cited)} line(s):"
    if len(cited) < len(indices):
        head = "\n".join(view.lines[i] for i in cited[:per_side])
        tail = "\n".join(view.lines[i] for i in cited[per_side:])
        body = f"{head}\n  ... (more lines follow between these) ...\n{tail}"
    else:
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
        "groups. Each group is a header naming the group followed by every line in it. "
        "Read ONLY the evidence below. Groups may hold very different numbers of lines -- "
        "that difference is not itself a sign of anomaly, so do not compare groups by which "
        "one has more lines; compare them by the timing, actors and event pattern the lines "
        "actually describe (e.g. how long a burst of activity spans, or whether errors recur "
        "without any corrective action). Never attribute a fact about one entity (an IP, a "
        "block id, a container id, a node) to a different entity, even one of the same kind "
        "-- two ids are two different things unless a line explicitly connects them.\n\n"
        f"Write an answer of at least {min_sentences} sentences that correlates the groups "
        "and explicitly uses facts from every group. End with one root-cause sentence, but "
        "keep it to what the timing and repetition pattern in the evidence itself supports "
        "-- 'these errors recur without a corrective action in between' is grounded; naming "
        "a specific unobserved cause ('a botnet', 'a misconfigured switch', 'a software bug') "
        "or a category of actor not shown in the evidence ('a script kiddie') is not, and "
        "must be avoided. Do not add sections, headers, bullet points, or a list of "
        "recommendations -- plain prose sentences only, nothing beyond the analysis asked "
        f"for.\n\nQUESTION: {question_text}\n\nEVIDENCE:\n{evidence_text}\n\nAnswer:"
    )


def build_hard_records(
    view: CorpusView,
    spec: DatasetSpec,
    config: GenerationConfig,
    client: VllmClient,
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
        client: vLLM client used for the draft and the groundedness check.

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
                    view, key, indices, params.evidence_lines_per_side, group_id
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
