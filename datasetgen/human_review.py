#!/usr/bin/env python3
"""
human_review.py

Human-review mechanism (Section 7.3 step 5 / Section 6): every in_review
record (medium- and hard-tier gold) needs a human accept/edit/reject
decision before it can become review_status=verified. This module never
makes that decision itself -- it only exports a worksheet for a human to
fill in, and applies the human's decisions back onto the dataset.

Called by main.py's `review-export` / `review-apply` subcommands.
"""
import csv
import json
import sys

WORKSHEET_FIELDS = ["id", "difficulty", "question", "draft_answer",
                     "groundedness_summary", "decision", "edited_answer"]


def _groundedness_summary(review_dir, qid):
    if not review_dir:
        return ""
    gpath = review_dir / f"{qid}.json"
    if not gpath.exists():
        return ""
    g = json.loads(gpath.read_text(encoding="utf-8"))
    supported = [c["supported"] for c in g.get("claims", [])]
    if not supported:
        return ""
    return f"{supported.count('yes')}/{len(supported)} claims supported"


def export(questions_path, worksheet_path, review_dir=None):
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        if q.get("review_status") != "in_review":
            continue
        rows.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "draft_answer": q["expected_answer"],
            "groundedness_summary": _groundedness_summary(review_dir, q["id"]),
            "decision": "",       # accept | edit | reject
            "edited_answer": "",  # only read when decision=edit
        })

    worksheet_path.parent.mkdir(parents=True, exist_ok=True)
    with open(worksheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} in_review record(s) to {worksheet_path}")
    return 0


def apply(questions_path, worksheet_path, out_path=None):
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}

    with open(worksheet_path, "r", encoding="utf-8") as f:
        decisions = list(csv.DictReader(f))

    accepted = edited = rejected = skipped = 0
    for row in decisions:
        qid = row["id"]
        q = by_id.get(qid)
        if q is None:
            print(f"  WARNING: worksheet id {qid} not found in questions file, skipping", file=sys.stderr)
            continue

        decision = (row.get("decision") or "").strip().lower()
        if decision == "accept":
            q["review_status"] = "verified"
            accepted += 1
        elif decision == "edit":
            edited_answer = (row.get("edited_answer") or "").strip()
            if not edited_answer:
                print(f"  WARNING: {qid} marked 'edit' but edited_answer is empty, leaving in_review",
                      file=sys.stderr)
                skipped += 1
            else:
                q["expected_answer"] = edited_answer
                q["review_status"] = "verified"
                edited += 1
        elif decision == "reject":
            q["review_status"] = "rejected"
            rejected += 1
        else:
            skipped += 1

    out_path = out_path or questions_path
    out_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Applied worksheet: accepted={accepted} edited={edited} rejected={rejected} "
          f"left_undecided={skipped}")
    print(f"Wrote updated dataset to {out_path}")
    return 0
