"""Easy tier (Section 7.1): count / lookup / presence.

The only tier with no model in it. Every gold value here is computed by SQL
against loghub's ``lines`` table and every one of them is recomputed by
``validate.py`` from the same table, which is why these records ship
``review_status=verified`` without a human step: nothing was asserted that a
reader cannot reproduce.

Three intents, and each is emitted under three phrasings of the same question, to
satisfy the Section 2/7.4 rule that a deterministic intent must be reachable
through at least three phrasing families. All three share one ``group_id``, so
they cannot be split across dev and test — a paraphrase of a test question
sitting in dev would be a leak.
"""

import sys
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec, LiteralSpec, LookupSpec
from src.params.generation_params import EasyTierParams, GenerationConfig
from src.utils import helper_postgres
from src.utils.helper_evidence import evidence_ref, gold_provenance, slugify

CREATED_BY = "src/generators/easy_tier.py@v1"

COUNT_PHRASINGS = (
    ("how-many", "How many log lines contain '{literal}'?"),
    ("count-the", "Count the lines mentioning '{literal}'."),
    ("total-of", "What is the total number of '{literal}' occurrences?"),
)

PRESENCE_PHRASINGS = (
    ("how-many", "Does the log contain any line with '{literal}'?"),
    ("count-the", "Check whether the log has a line mentioning '{literal}'."),
    ("imperative", "Tell me if there is at least one occurrence of '{literal}' in the log."),
)

LOOKUP_PHRASINGS = {
    "first": (
        ("how-many", "Show the first line that reports '{literal}'."),
        ("count-the", "Find the first log line mentioning '{literal}'."),
        ("imperative", "Give me the first line where '{literal}' appears."),
    ),
    "last": (
        ("how-many", "Show the last line that reports '{literal}'."),
        ("count-the", "Find the last log line mentioning '{literal}'."),
        ("imperative", "Give me the last line where '{literal}' appears."),
    ),
}


def count_matches(dataset_key: str, literal: str, case_sensitive: bool) -> tuple[int, list[int]]:
    """Counts a literal's matching lines via Postgres.

    Args:
        dataset_key: Lowercase dataset key.
        literal: Substring to match.
        case_sensitive: Whether matching is case sensitive.

    Returns:
        Tuple ``(count, matched_indices)`` where the indices are 0-based
        positions into ``CorpusView.lines`` — the SQL returns 1-based line
        numbers, and converting here keeps evidence citation at the call sites
        plain list indexing.
    """
    count, line_numbers = helper_postgres.count_literal(
        dataset_key, literal, case_sensitive
    )
    return count, [ln - 1 for ln in line_numbers]


def _build_count_or_presence(
    view: CorpusView,
    literal_spec: LiteralSpec,
    kind: str,
    config: GenerationConfig,
    params: EasyTierParams,
) -> list[dict[str, Any]] | None:
    """Builds the three phrasings of one count or presence question.

    A count literal with too few matches is pruned: a question whose answer is
    "1" tests nothing about aggregation and its answer is unstable under any
    corpus change. A *presence* literal with zero matches is kept, because "No"
    is exactly the answer that case is curated to produce — and since there is no
    matching line to cite, its evidence anchors on line 1, which still resolves
    to a real hash-verifiable line from the same corpus (the schema requires at
    least one ref).

    Args:
        view: The dataset's corpus.
        literal_spec: The literal to count or test for.
        kind: ``"count"`` or ``"presence"``.
        config: Generation config, for the provenance stamp.
        params: Easy-tier knobs.

    Returns:
        Three records, or ``None`` when the literal was pruned.
    """
    literal = literal_spec.literal
    count, matched_indices = count_matches(view.key, literal, literal_spec.case_sensitive)

    if kind == "count" and count < params.min_matches:
        print(
            f"  [prune] {view.name}: count literal '{literal}' has only {count} match(es) "
            f"(<{params.min_matches}), skipping",
            file=sys.stderr,
        )
        return None

    slug = slugify(literal)
    group_id = f"{view.key}:{kind}:{slug}"

    cited_indices = matched_indices[: params.max_cited_lines]
    if kind == "presence" and count == 0:
        cited_indices = [0]
    refs = [evidence_ref(view.key, i + 1, view.lines[i], group_id) for i in cited_indices]

    if kind == "presence":
        answer_text = "Yes" if count > 0 else "No"
        answer_type = "presence"
        phrasings = PRESENCE_PHRASINGS
    else:
        answer_text = str(count)
        answer_type = "count"
        phrasings = COUNT_PHRASINGS

    records = []
    for idx, (family, template) in enumerate(phrasings):
        records.append(
            {
                "id": f"{view.key}_v1_{kind}_{slug}_{idx}",
                "question": template.format(literal=literal),
                "routing_path": "sql",
                "answer_type": answer_type,
                "task": "Aggregation",
                "difficulty": "easy",
                "phrasing_family": family,
                "review_status": "verified",
                "reviewers": [config.reviewer],
                "expected_answer": answer_text,
                "gold_provenance": gold_provenance(
                    method="deterministic_aggregation",
                    created_by=CREATED_BY,
                    created_at=config.created_at,
                    corpus_sha256=view.sha256,
                ),
                "numeric_claims": [
                    {
                        "value": count,
                        "query": {
                            "operator": "count_literal",
                            "literal": literal,
                            "case_sensitive": literal_spec.case_sensitive,
                        },
                    }
                ],
                "evidence": {"refs": refs},
            }
        )
    return records


def _build_lookup(
    view: CorpusView, lookup_spec: LookupSpec, config: GenerationConfig
) -> list[dict[str, Any]] | None:
    """Builds the three phrasings of one first/last line-lookup question.

    Args:
        view: The dataset's corpus.
        lookup_spec: The literal and whether the first or last match is wanted.
        config: Generation config, for the provenance stamp.

    Returns:
        Three records, or ``None`` when the literal matches nothing — unlike
        presence, a lookup with no match has no answer to give.
    """
    literal = lookup_spec.literal
    count, matched_indices = count_matches(view.key, literal, lookup_spec.case_sensitive)
    if count < 1:
        print(
            f"  [prune] {view.name}: lookup literal '{literal}' has 0 matches, skipping",
            file=sys.stderr,
        )
        return None

    idx = matched_indices[0] if lookup_spec.position == "first" else matched_indices[-1]
    line_number = idx + 1
    line_text = view.lines[idx]

    slug = slugify(literal) + "_" + lookup_spec.position
    group_id = f"{view.key}:lookup:{slug}"
    refs = [evidence_ref(view.key, line_number, line_text, group_id)]

    records = []
    for i, (family, template) in enumerate(LOOKUP_PHRASINGS[lookup_spec.position]):
        records.append(
            {
                "id": f"{view.key}_v1_lookup_{slug}_{i}",
                "question": template.format(literal=literal),
                "routing_path": "keyword",
                "answer_type": "line_lookup",
                "task": "Lookup",
                "difficulty": "easy",
                "phrasing_family": family,
                "review_status": "verified",
                "reviewers": [config.reviewer],
                "expected_answer": line_text,
                "gold_provenance": gold_provenance(
                    method="deterministic_aggregation",
                    created_by=CREATED_BY,
                    created_at=config.created_at,
                    corpus_sha256=view.sha256,
                ),
                "evidence": {"refs": refs},
            }
        )
    return records


def build_easy_records(
    view: CorpusView, spec: DatasetSpec, config: GenerationConfig
) -> list[dict[str, Any]]:
    """Builds every easy-tier record for one dataset.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        config: Generation config; its ``easy`` field carries the tier knobs.

    Returns:
        All count, presence and lookup records that survived pruning.
    """
    records: list[dict[str, Any]] = []
    for literal_spec in spec.count_literals:
        built = _build_count_or_presence(view, literal_spec, "count", config, config.easy)
        if built:
            records.extend(built)
    for literal_spec in spec.presence_literals:
        built = _build_count_or_presence(view, literal_spec, "presence", config, config.easy)
        if built:
            records.extend(built)
    for lookup_spec in spec.lookup_specs:
        built = _build_lookup(view, lookup_spec, config)
        if built:
            records.extend(built)
    return records


def select_official_20(
    easy_records: list[dict[str, Any]], dataset_keys: list[str]
) -> list[dict[str, Any]]:
    """Narrows the full easy tier to the official 20-question stage-1 set.

    Exactly one count and one presence question per LogHub dataset, first
    phrasing only, no lookups. Deliberately narrower than the full easy tier:
    this is the set that stays the official output until human-reviewed
    medium/hard questions are folded in, growing the 20 upward per the staged
    scaling plan (Section 3.1).

    Args:
        easy_records: Every easy-tier record produced this pass.
        dataset_keys: Dataset keys, in the order they should appear.

    Returns:
        The selected records, ordered by dataset then count-before-presence.
    """
    by_id = {r["id"]: r for r in easy_records}
    selected = []
    for key in dataset_keys:
        for kind in ("count", "presence"):
            match = next(
                (
                    rid
                    for rid in by_id
                    if rid.startswith(f"{key}_v1_{kind}_") and rid.endswith("_0")
                ),
                None,
            )
            if match:
                selected.append(by_id[match])
    return selected
