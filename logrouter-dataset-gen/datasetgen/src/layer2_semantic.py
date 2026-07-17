#!/usr/bin/env python3
"""
layer2_semantic.py

Section 7.2 - Tier 2 (medium): single-event-family explanation/summary,
<=8 evidence lines, 1-3 sentence gold drafted by nemotron-3-nano:30b
(Section 5.5). These records always start life as review_status=in_review;
a human must verify (or edit/reject) them before they count as usable gold
(Section 6, Bilimsel Durustluk Kurali).

Usage:
  python3 layer2_semantic.py --corpus-dir /data/loghub --out /output/pilot/layer2.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

from corpus import dataset_key, load_lines, sha256_bytes, sha256_line
from dataset_specs import DATASET_SPECS
from ollama_client import OllamaClient
from split import split_for_group

REVIEWER = "faz1_pilot_script"
CREATED_AT = "2026-08-01T00:00:00Z"  # fixed for determinism (Section 6)
WINDOW_SIZE = 8
QUESTIONS_PER_DATASET = 2


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _find_matches(lines, literal):
    needle = literal.lower()
    return [i for i, line in enumerate(lines) if needle in line.lower()]


def _evidence_window(lines, start_idx, size=WINDOW_SIZE):
    end_idx = min(len(lines), start_idx + size)
    return list(range(start_idx, end_idx))


def _evidence_ref(dkey, line_number, line_text, group_id):
    h = sha256_line(line_text)
    return {
        "id": f"{dkey}:line:{line_number:08d}:{h[7:23]}",
        "line_number": line_number,
        "line_hash": h,
        "group_id": group_id,
    }


def build_prompt(dataset_name, evidence_lines):
    numbered = "\n".join(evidence_lines)
    return (
        f"You are analyzing a short excerpt of raw {dataset_name} system log lines. "
        "Read ONLY the evidence below and answer using nothing but what these lines show "
        "-- do not speculate about anything not directly stated. In 1 to 3 sentences, "
        "explain what event or condition this excerpt shows and what it means.\n\n"
        f"EVIDENCE:\n{numbered}\n\n"
        "Answer with only the explanation (1-3 sentences), no preamble, no bullet points."
    )


def generate_for_dataset(spec, corpus_dir: Path, client: OllamaClient):
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

    # Spread the QUESTIONS_PER_DATASET occurrence picks across the file so their
    # evidence windows don't overlap.
    step = max(1, len(match_indices) // QUESTIONS_PER_DATASET)
    picks = []
    for i in range(0, len(match_indices), step):
        picks.append(match_indices[i])
        if len(picks) >= QUESTIONS_PER_DATASET:
            break

    slug = _slugify(spec.medium_anchor_literal)
    records = []
    for occ_num, start_idx in enumerate(picks):
        window_indices = _evidence_window(lines, start_idx)
        evidence_lines = [lines[i] for i in window_indices]
        group_id = f"{dkey}:semantic:{slug}_{occ_num}"
        split = split_for_group(group_id)
        refs = [_evidence_ref(dkey, i + 1, lines[i], group_id) for i in window_indices]

        prompt = build_prompt(spec.name, evidence_lines)
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
                "created_by": "datasetgen/layer2_semantic.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
                "model": draft["model"],
            },
            "evidence": {"refs": refs},
        })
        print(f"  [{spec.name}] drafted semantic question {occ_num} "
              f"({len(evidence_lines)} evidence lines)")
    return records


def generate_all(corpus_dir: Path, client: OllamaClient):
    all_records = []
    for name, spec in DATASET_SPECS.items():
        recs = generate_for_dataset(spec, corpus_dir, client)
        print(f"[{name}] layer2: {len(recs)} questions")
        all_records.extend(recs)
    return all_records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-parallel-model-calls", type=int, default=4)
    args = ap.parse_args()

    client = OllamaClient(max_parallel_calls=args.max_parallel_model_calls)
    records = generate_all(args.corpus_dir, client)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} layer-2 (medium) questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
