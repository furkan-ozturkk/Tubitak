#!/usr/bin/env python3
"""
layer1_deterministic.py

Section 7.1 - Tier 1 (easy): count / lookup / presence. Gold values are
computed directly from the corpus; no model call is involved, so these
records are self-verifying (review_status=verified). Every deterministic
intent is written with >=3 phrasing families sharing one group_id
(Section 2/7.4).

Usage:
  python3 layer1_deterministic.py --corpus-dir /data/loghub --out /output/pilot/layer1.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

from corpus import dataset_key, load_lines, sha256_bytes, sha256_line
from dataset_specs import DATASET_SPECS
from split import split_for_group

REVIEWER = "faz1_pilot_script"
CREATED_AT = "2026-08-01T00:00:00Z"  # fixed for determinism (Section 6)
MIN_MATCHES = 3
MAX_CITED_LINES = 5

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


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _count_matches(lines, literal, case_sensitive):
    needle = literal if case_sensitive else literal.lower()
    matched_indices = []
    for i, line in enumerate(lines):
        hay = line if case_sensitive else line.lower()
        if needle in hay:
            matched_indices.append(i)
    return len(matched_indices), matched_indices


def _evidence_ref(dkey, line_number, line_text, group_id):
    h = sha256_line(line_text)
    return {
        "id": f"{dkey}:line:{line_number:08d}:{h[7:23]}",
        "line_number": line_number,
        "line_hash": h,
        "group_id": group_id,
    }


def generate_count_or_presence(dataset_name, dkey, lines, literal_spec, kind, corpus_sha256):
    """kind: 'count' or 'presence'. Returns a list of records (one per phrasing) or None if pruned."""
    literal = literal_spec.literal
    count, matched_indices = _count_matches(lines, literal, literal_spec.case_sensitive)

    if kind == "count" and count < MIN_MATCHES:
        print(f"  [prune] {dataset_name}: count literal '{literal}' has only {count} match(es) "
              f"(<{MIN_MATCHES}), skipping", file=sys.stderr)
        return None
    # Presence literals may legitimately have 0 matches -- that is a valid "No" example.

    slug = _slugify(literal)
    group_id = f"{dkey}:{kind}:{slug}"
    split = split_for_group(group_id)

    cited_indices = matched_indices[:MAX_CITED_LINES]
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
                "created_by": "datasetgen/layer1_deterministic.py@v1",
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


def generate_lookup(dataset_name, dkey, lines, lookup_spec, corpus_sha256):
    literal = lookup_spec.literal
    count, matched_indices = _count_matches(lines, literal, lookup_spec.case_sensitive)
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
                "created_by": "datasetgen/layer1_deterministic.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
            },
            "evidence": {"refs": refs},
        })
    return records


def process_dataset(spec, corpus_dir: Path):
    log_path = corpus_dir / spec.log_filename
    data = log_path.read_bytes()
    corpus_sha256 = sha256_bytes(data)
    lines = load_lines(log_path)
    dkey = dataset_key(spec.name)

    records = []
    for lit in spec.count_literals:
        recs = generate_count_or_presence(spec.name, dkey, lines, lit, "count", corpus_sha256)
        if recs:
            records.extend(recs)
    for lit in spec.presence_literals:
        recs = generate_count_or_presence(spec.name, dkey, lines, lit, "presence", corpus_sha256)
        if recs:
            records.extend(recs)
    for lk in spec.lookup_specs:
        recs = generate_lookup(spec.name, dkey, lines, lk, corpus_sha256)
        if recs:
            records.extend(recs)
    return records


def generate_all(corpus_dir: Path):
    all_records = []
    for name, spec in DATASET_SPECS.items():
        recs = process_dataset(spec, corpus_dir)
        print(f"[{name}] layer1: {len(recs)} questions")
        all_records.extend(recs)
    return all_records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = generate_all(args.corpus_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} layer-1 (easy) questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
