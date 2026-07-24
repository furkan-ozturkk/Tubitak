#!/usr/bin/env python3
"""
question_generators.py

The three question-generation tiers of Section 7, merged into one module
(they used to be layer1_deterministic.py / layer2_semantic.py /
layer3_hard.py):

  - generate_easy   (Section 7.1): count / lookup / presence. Computed
    directly from the corpus, no model call, self-verifying
    (review_status=verified).
  - generate_medium (Section 7.2): single-event-family explanation/summary,
    drafted by nemotron-3-nano:30b (Section 5.5). review_status=in_review
    until a human accepts/edits it (see human_review.py).
  - generate_hard    (Section 7.3): multi-event / root-cause / synthesis.
    Selects >=2 evidence groups first, drafts with nemotron-3-nano:30b, then
    runs a claim-by-claim groundedness check with gpt-oss:20b (a different
    model family, Section 5.5/6) and saves it to review/groundedness/<id>.json.
    review_status=in_review until a human accepts/edits it.

Called by main.py's `generate` subcommand.
"""
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pg_client
from corpus_utils import dataset_key, load_lines, sha256_bytes, sha256_line, split_for_group
from loghub_datasets import DATASET_SPECS
from ollama_client import OllamaClient

REVIEWER = "faz1_pilot_script"
CREATED_AT = "2026-08-01T00:00:00Z"  # fixed for determinism (Section 6)


@dataclass(frozen=True)
class EasyTierParams:
    """Tuning knobs for generate_easy (Section 7.1). Not exposed on the CLI --
    these are curation-level constants, not something an operator scales."""
    min_matches: int = 3       # count literals with fewer real matches are pruned
    max_cited_lines: int = 5   # evidence.refs cap per count/presence question


@dataclass(frozen=True)
class MediumTierParams:
    """Tuning knobs for generate_medium (Section 7.2)."""
    window_size: int = 8              # evidence lines per medium question (<=8, Section 7.2)
    questions_per_dataset: int = 2    # occurrences drafted per dataset


@dataclass(frozen=True)
class HardTierParams:
    """Tuning knobs for generate_hard (Section 7.3)."""
    min_sentences: int = 4    # gold answers must be >=4 sentences (Section 7.3)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _evidence_ref(dkey, line_number, line_text, group_id):
    h = sha256_line(line_text)
    return {
        "id": f"{dkey}:line:{line_number:08d}:{h[7:23]}",
        "line_number": line_number,
        "line_hash": h,
        "group_id": group_id,
    }


# --------------------------------------------------------------------------
# Tier 1 (easy): count / lookup / presence
# --------------------------------------------------------------------------
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


def _count_matches(dkey, literal, case_sensitive):
    """Queries loghub's Postgres `lines` table (pg_client.py) instead of scanning
    an in-memory list. Returns (count, matched_indices), matched_indices being
    0-based positions -- same contract callers already expect from the old
    file-scanning version, so evidence citation (lines[i]) still works unchanged."""
    count, line_numbers = pg_client.count_literal(dkey, literal, case_sensitive)
    matched_indices = [ln - 1 for ln in line_numbers]
    return count, matched_indices


def _generate_count_or_presence(dataset_name, dkey, lines, literal_spec, kind, corpus_sha256,
                                 params: EasyTierParams):
    """kind: 'count' or 'presence'. Returns a list of records (one per phrasing) or None if pruned."""
    literal = literal_spec.literal
    count, matched_indices = _count_matches(dkey, literal, literal_spec.case_sensitive)

    if kind == "count" and count < params.min_matches:
        print(f"  [prune] {dataset_name}: count literal '{literal}' has only {count} match(es) "
              f"(<{params.min_matches}), skipping", file=sys.stderr)
        return None
    # Presence literals may legitimately have 0 matches -- that is a valid "No" example.

    slug = _slugify(literal)
    group_id = f"{dkey}:{kind}:{slug}"
    split = split_for_group(group_id)

    cited_indices = matched_indices[:params.max_cited_lines]
    if kind == "presence" and count == 0:
        # No matching line exists to cite for a true negative; anchor evidence on line 1
        # instead so evidence.refs (schema-required, minItems=1) still points at a real,
        # hash-verifiable line from this corpus file.
        cited_indices = [0]
    refs = [_evidence_ref(dkey, i + 1, lines[i], group_id) for i in cited_indices]

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
        records.append({
            "id": f"{dkey}_v1_{kind}_{slug}_{idx}",
            "question": template.format(literal=literal),
            "routing_path": "sql",
            "answer_type": answer_type,
            "task": "Aggregation",
            "difficulty": "easy",
            "phrasing_family": family,
            "split": split,
            "review_status": "verified",
            "reviewers": [REVIEWER],
            "expected_answer": answer_text,
            "gold_provenance": {
                "method": "deterministic_aggregation",
                "created_by": "datasetgen/question_generators.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
            },
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
        })
    return records


def _generate_lookup(dataset_name, dkey, lines, lookup_spec, corpus_sha256):
    literal = lookup_spec.literal
    count, matched_indices = _count_matches(dkey, literal, lookup_spec.case_sensitive)
    if count < 1:
        print(f"  [prune] {dataset_name}: lookup literal '{literal}' has 0 matches, skipping",
              file=sys.stderr)
        return None

    idx = matched_indices[0] if lookup_spec.position == "first" else matched_indices[-1]
    line_number = idx + 1
    line_text = lines[idx]

    slug = _slugify(literal) + "_" + lookup_spec.position
    group_id = f"{dkey}:lookup:{slug}"
    split = split_for_group(group_id)
    refs = [_evidence_ref(dkey, line_number, line_text, group_id)]

    records = []
    for i, (family, template) in enumerate(LOOKUP_PHRASINGS[lookup_spec.position]):
        records.append({
            "id": f"{dkey}_v1_lookup_{slug}_{i}",
            "question": template.format(literal=literal),
            "routing_path": "keyword",
            "answer_type": "line_lookup",
            "task": "Lookup",
            "difficulty": "easy",
            "phrasing_family": family,
            "split": split,
            "review_status": "verified",
            "reviewers": [REVIEWER],
            "expected_answer": line_text,
            "gold_provenance": {
                "method": "deterministic_aggregation",
                "created_by": "datasetgen/question_generators.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
            },
            "evidence": {"refs": refs},
        })
    return records


def _process_easy_dataset(spec, corpus_dir: Path, params: EasyTierParams):
    log_path = corpus_dir / spec.log_filename
    data = log_path.read_bytes()
    corpus_sha256 = sha256_bytes(data)
    lines = load_lines(log_path)
    dkey = dataset_key(spec.name)

    records = []
    for lit in spec.count_literals:
        recs = _generate_count_or_presence(spec.name, dkey, lines, lit, "count", corpus_sha256, params)
        if recs:
            records.extend(recs)
    for lit in spec.presence_literals:
        recs = _generate_count_or_presence(spec.name, dkey, lines, lit, "presence", corpus_sha256, params)
        if recs:
            records.extend(recs)
    for lk in spec.lookup_specs:
        recs = _generate_lookup(spec.name, dkey, lines, lk, corpus_sha256)
        if recs:
            records.extend(recs)
    return records


def generate_easy(corpus_dir: Path, params: EasyTierParams = EasyTierParams()):
    all_records = []
    for name, spec in DATASET_SPECS.items():
        recs = _process_easy_dataset(spec, corpus_dir, params)
        print(f"[{name}] easy: {len(recs)} questions")
        all_records.extend(recs)
    return all_records


def generate_official_20(corpus_dir: Path, params: EasyTierParams = EasyTierParams()):
    """The default `main.py generate` output (Section 3.1 stage 1): exactly
    1 count + 1 presence question per LogHub dataset (first phrasing only,
    no lookup), no model involved. Deliberately narrower than generate_easy()
    (which produces the full easy tier with every phrasing) -- this is what
    stays the official output/pilot/questions.json until medium/hard
    questions get folded in via human review (see main.py's --full)."""
    all_easy = generate_easy(corpus_dir, params)
    by_id = {r["id"]: r for r in all_easy}

    selected = []
    for name, spec in DATASET_SPECS.items():
        dkey = dataset_key(name)
        count_id = next((rid for rid in by_id if rid.startswith(f"{dkey}_v1_count_") and rid.endswith("_0")), None)
        presence_id = next((rid for rid in by_id if rid.startswith(f"{dkey}_v1_presence_") and rid.endswith("_0")), None)
        if count_id:
            selected.append(by_id[count_id])
        if presence_id:
            selected.append(by_id[presence_id])
    return selected


# --------------------------------------------------------------------------
# Tier 2 (medium): single-event-family explanation/summary
# --------------------------------------------------------------------------
def _find_matches(lines, literal):
    needle = literal.lower()
    return [i for i, line in enumerate(lines) if needle in line.lower()]


def _evidence_window(lines, start_idx, size):
    end_idx = min(len(lines), start_idx + size)
    return list(range(start_idx, end_idx))


def _build_medium_prompt(dataset_name, evidence_lines):
    numbered = "\n".join(evidence_lines)
    return (
        f"You are analyzing a short excerpt of raw {dataset_name} system log lines. "
        "Read ONLY the evidence below and answer using nothing but what these lines show "
        "-- do not speculate about anything not directly stated. In 1 to 3 sentences, "
        "explain what event or condition this excerpt shows and what it means.\n\n"
        f"EVIDENCE:\n{numbered}\n\n"
        "Answer with only the explanation (1-3 sentences), no preamble, no bullet points."
    )


def _generate_medium_for_dataset(spec, corpus_dir: Path, client: OllamaClient, params: MediumTierParams):
    log_path = corpus_dir / spec.log_filename
    data = log_path.read_bytes()
    corpus_sha256 = sha256_bytes(data)
    lines = load_lines(log_path)
    dkey = dataset_key(spec.name)

    if not spec.medium_anchor_literal:
        return []

    match_indices = _find_matches(lines, spec.medium_anchor_literal)
    if not match_indices:
        print(f"  [prune] {spec.name}: medium anchor '{spec.medium_anchor_literal}' has 0 matches, "
              f"skipping", file=sys.stderr)
        return []

    # Spread the questions_per_dataset occurrence picks across the file so their
    # evidence windows don't overlap.
    step = max(1, len(match_indices) // params.questions_per_dataset)
    picks = []
    for i in range(0, len(match_indices), step):
        picks.append(match_indices[i])
        if len(picks) >= params.questions_per_dataset:
            break

    slug = _slugify(spec.medium_anchor_literal)
    records = []
    for occ_num, start_idx in enumerate(picks):
        window_indices = _evidence_window(lines, start_idx, params.window_size)
        evidence_lines = [lines[i] for i in window_indices]
        group_id = f"{dkey}:semantic:{slug}_{occ_num}"
        split = split_for_group(group_id)
        refs = [_evidence_ref(dkey, i + 1, lines[i], group_id) for i in window_indices]

        prompt = _build_medium_prompt(spec.name, evidence_lines)
        draft = client.draft(prompt, temperature=0.0)

        records.append({
            "id": f"{dkey}_v1_semantic_{slug}_{occ_num}",
            "question": f"Looking at these {spec.name} log lines, what happened and what does it mean?",
            "routing_path": "semantic",
            "answer_type": "explanation",
            "task": "Summarization",
            "difficulty": "medium",
            "phrasing_family": "single-event-summary",
            "split": split,
            "review_status": "in_review",
            "reviewers": [REVIEWER],
            "expected_answer": draft["text"],
            "gold_provenance": {
                "method": "independent_model_then_human",
                "created_by": "datasetgen/question_generators.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
                "model": draft["model"],
            },
            "evidence": {"refs": refs},
        })
        print(f"  [{spec.name}] drafted semantic question {occ_num} "
              f"({len(evidence_lines)} evidence lines)")
    return records


def generate_medium(corpus_dir: Path, client: OllamaClient, params: MediumTierParams = MediumTierParams()):
    all_records = []
    for name, spec in DATASET_SPECS.items():
        recs = _generate_medium_for_dataset(spec, corpus_dir, client, params)
        print(f"[{name}] medium: {len(recs)} questions")
        all_records.extend(recs)
    return all_records


# --------------------------------------------------------------------------
# Tier 3 (hard): multi-event / root-cause / synthesis
# --------------------------------------------------------------------------


def _select_hard_groups(lines, hard_spec):
    pattern = re.compile(hard_spec.extract_key_regex)
    by_key = defaultdict(list)
    for i, line in enumerate(lines):
        m = pattern.search(line)
        if m:
            by_key[m.group("key")].append(i)

    qualifying = {k: idxs for k, idxs in by_key.items() if len(idxs) >= hard_spec.min_lines_per_group}
    if len(qualifying) < hard_spec.num_groups:
        return None

    # Deterministic ranking: most lines first, key name breaks ties.
    ranked = sorted(qualifying.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return ranked[: hard_spec.num_groups]


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _build_hard_evidence_block(dkey, lines, key, indices, hard_spec, group_id):
    cited = indices[: hard_spec.evidence_lines_per_group]
    refs = [_evidence_ref(dkey, i + 1, lines[i], group_id) for i in cited]
    text_block = f"[Group: {key}]\n" + "\n".join(lines[i] for i in cited)
    return refs, text_block


def _generate_hard_for_dataset(spec, corpus_dir: Path, client: OllamaClient, review_dir: Path,
                                params: HardTierParams):
    if not spec.hard_groups:
        return []

    log_path = corpus_dir / spec.log_filename
    data = log_path.read_bytes()
    corpus_sha256 = sha256_bytes(data)
    lines = load_lines(log_path)
    dkey = dataset_key(spec.name)

    records = []
    for hard_spec in spec.hard_groups:
        selected = _select_hard_groups(lines, hard_spec)
        if selected is None:
            print(f"  [quota_unmet] {spec.name}/{hard_spec.spec_id}: fewer than "
                  f"{hard_spec.num_groups} groups with >={hard_spec.min_lines_per_group} lines",
                  file=sys.stderr)
            continue

        keys = [k for k, _ in selected]
        group_ids = [f"{dkey}:hard:{hard_spec.spec_id}:{_slugify(k)}" for k in keys]
        all_refs = []
        evidence_blocks = []
        for (key, indices), group_id in zip(selected, group_ids):
            refs, block = _build_hard_evidence_block(dkey, lines, key, indices, hard_spec, group_id)
            all_refs.extend(refs)
            evidence_blocks.append(block)
        evidence_text = "\n\n".join(evidence_blocks)

        question_text = hard_spec.question_template.format(**{f"key{i}": k for i, k in enumerate(keys)})

        prompt = (
            f"You are analyzing raw {spec.name} log lines from {len(keys)} related event groups. "
            "Read ONLY the evidence below; do not speculate about anything not shown. Write an "
            f"answer of at least {params.min_sentences} sentences that correlates the groups, explicitly "
            "uses facts from every group, and proposes a root-cause hypothesis.\n\n"
            f"QUESTION: {question_text}\n\nEVIDENCE:\n{evidence_text}\n\nAnswer:"
        )
        draft = client.draft(prompt, temperature=0.0)
        answer_text = draft["text"]

        claims = _split_sentences(answer_text)
        claim_records = []
        for claim in claims:
            verdict, _raw = client.groundedness_check(claim, evidence_text)
            claim_records.append({
                "text": claim,
                "supported": verdict,
                "evidence_ref": all_refs[0]["id"] if verdict != "no" else None,
            })

        qid = f"{dkey}_v1_hard_{hard_spec.spec_id}"
        groundedness_report = {
            "question_id": qid,
            "claims": claim_records,
            "verdict": "needs_human_review",
        }
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{qid}.json").write_text(
            json.dumps(groundedness_report, ensure_ascii=False, indent=2), encoding="utf-8")

        # A hard question's own split is anchored on its first group_id; every ref across
        # both groups is still part of one evidence-linked question record.
        split = split_for_group(group_ids[0])

        records.append({
            "id": qid,
            "question": question_text,
            "routing_path": "semantic",
            "answer_type": "synthesis",
            "task": hard_spec.task,
            "difficulty": "hard",
            "phrasing_family": "hard-synthesis",
            "split": split,
            "review_status": "in_review",
            "reviewers": [REVIEWER],
            "expected_answer": answer_text,
            "gold_provenance": {
                "method": "independent_model_then_human",
                "created_by": "datasetgen/question_generators.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
                "model": draft["model"],
            },
            "evidence": {"refs": all_refs},
        })
        print(f"  [{spec.name}] drafted hard question '{hard_spec.spec_id}' "
              f"({len(claims)} claims, groups={keys})")
    return records


def generate_hard(corpus_dir: Path, client: OllamaClient, review_dir: Path,
                   params: HardTierParams = HardTierParams()):
    all_records = []
    for name, spec in DATASET_SPECS.items():
        recs = _generate_hard_for_dataset(spec, corpus_dir, client, review_dir, params)
        print(f"[{name}] hard: {len(recs)} questions")
        all_records.extend(recs)
    return all_records
