#!/usr/bin/env python3
"""
layer3_hard.py

Section 7.3 - Tier 3 (hard): multi-event / root-cause / synthesis. Follows
the hard-gold protocol in order:
  1. Select evidence groups first (>=2 distinct group_ids per question).
  2. The question is written from the evidence (never the other way round).
  3. nemotron-3-nano:30b drafts the gold answer, seeing ONLY those evidence
     lines, temperature=0, thinking mode off.
  4. gpt-oss:20b (a different model family, Section 5.5/6) runs a
     groundedness check claim-by-claim and the result is saved to
     review/groundedness/<id>.json in the same format as Section 7.3.
  5. Human review (accept/edit/reject) happens afterwards, outside this
     script -- every record this script emits is review_status=in_review.

Simplification note: step 4 ideally checks each claim only against the
evidence lines it actually refers to; for the pilot every claim is checked
against the full (both-group) evidence text instead, since automatically
attributing a claim to "its" group is its own NLP problem. Worth revisiting
in Phase 2.

Usage:
  python3 layer3_hard.py --corpus-dir /data/loghub --out /output/pilot/layer3.json \
      --review-dir /output/pilot/review/groundedness
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from corpus import dataset_key, load_lines, sha256_bytes, sha256_line
from dataset_specs import DATASET_SPECS
from ollama_client import OllamaClient
from split import split_for_group

REVIEWER = "faz1_pilot_script"
CREATED_AT = "2026-08-01T00:00:00Z"  # fixed for determinism (Section 6)
MIN_SENTENCES = 4


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


def _select_groups(lines, hard_spec):
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


def _build_evidence_block(dkey, lines, key, indices, hard_spec, group_id):
    cited = indices[: hard_spec.evidence_lines_per_group]
    refs = [_evidence_ref(dkey, i + 1, lines[i], group_id) for i in cited]
    text_block = f"[Group: {key}]\n" + "\n".join(lines[i] for i in cited)
    return refs, text_block


def generate_for_dataset(spec, corpus_dir: Path, client: OllamaClient, review_dir: Path):
    if not spec.hard_groups:
        return []

    log_path = corpus_dir / spec.log_filename
    data = log_path.read_bytes()
    corpus_sha256 = sha256_bytes(data)
    lines = load_lines(log_path)
    dkey = dataset_key(spec.name)

    records = []
    for hard_spec in spec.hard_groups:
        selected = _select_groups(lines, hard_spec)
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
            refs, block = _build_evidence_block(dkey, lines, key, indices, hard_spec, group_id)
            all_refs.extend(refs)
            evidence_blocks.append(block)
        evidence_text = "\n\n".join(evidence_blocks)

        question_text = hard_spec.question_template.format(**{f"key{i}": k for i, k in enumerate(keys)})

        prompt = (
            f"You are analyzing raw {spec.name} log lines from {len(keys)} related event groups. "
            "Read ONLY the evidence below; do not speculate about anything not shown. Write an "
            f"answer of at least {MIN_SENTENCES} sentences that correlates the groups, explicitly "
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
                "created_by": "datasetgen/layer3_hard.py@v1",
                "created_at": CREATED_AT,
                "corpus_sha256": corpus_sha256,
                "model": draft["model"],
            },
            "evidence": {"refs": all_refs},
        })
        print(f"  [{spec.name}] drafted hard question '{hard_spec.spec_id}' "
              f"({len(claims)} claims, groups={keys})")
    return records


def generate_all(corpus_dir: Path, client: OllamaClient, review_dir: Path):
    all_records = []
    for name, spec in DATASET_SPECS.items():
        recs = generate_for_dataset(spec, corpus_dir, client, review_dir)
        print(f"[{name}] layer3: {len(recs)} questions")
        all_records.extend(recs)
    return all_records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review-dir", type=Path, required=True)
    ap.add_argument("--max-parallel-model-calls", type=int, default=4)
    args = ap.parse_args()

    client = OllamaClient(max_parallel_calls=args.max_parallel_model_calls)
    records = generate_all(args.corpus_dir, client, args.review_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} layer-3 (hard) questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
