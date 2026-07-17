#!/usr/bin/env python3
"""
review_worksheet.py

Human-review mechanism (Section 7.3 step 5 / Section 6): every in_review
record (medium- and hard-tier gold) needs a human accept/edit/reject
decision before it can become review_status=verified. This script never
makes that decision itself -- it only exports a worksheet for a human to
fill in, and applies the human's decisions back onto the dataset.

Usage:
  # 1) export the in_review records to a worksheet:
  python3 review_worksheet.py export --questions /output/pilot/questions.json \
      --worksheet /output/pilot/review/worksheet.csv \
      --review-dir /output/pilot/review/groundedness

  # 2) a human fills in the "decision" column with accept | edit | reject,
  #    and "edited_answer" if decision=edit

  # 3) apply the filled-in worksheet back onto the dataset:
  python3 review_worksheet.py apply --questions /output/pilot/questions.json \
      --worksheet /output/pilot/review/worksheet.csv --out /output/pilot/questions.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

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


def cmd_export(args):
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    rows = []
    for q in questions:
        if q.get("review_status") != "in_review":
            continue
        rows.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "draft_answer": q["expected_answer"],
            "groundedness_summary": _groundedness_summary(args.review_dir, q["id"]),
            "decision": "",       # accept | edit | reject
            "edited_answer": "",  # only read when decision=edit
        })

    args.worksheet.parent.mkdir(parents=True, exist_ok=True)
    with open(args.worksheet, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WORKSHEET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} in_review record(s) to {args.worksheet}")
    return 0


def cmd_apply(args):
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in questions}

    with open(args.worksheet, "r", encoding="utf-8") as f:
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

    out_path = args.out or args.questions
    out_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Applied worksheet: accepted={accepted} edited={edited} rejected={rejected} "
          f"left_undecided={skipped}")
    print(f"Wrote updated dataset to {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="export in_review records to a CSV worksheet")
    export_p.add_argument("--questions", type=Path, required=True)
    export_p.add_argument("--worksheet", type=Path, required=True)
    export_p.add_argument("--review-dir", type=Path, default=None)
    export_p.set_defaults(func=cmd_export)

    apply_p = sub.add_parser("apply", help="apply a filled-in worksheet back onto the questions file")
    apply_p.add_argument("--questions", type=Path, required=True)
    apply_p.add_argument("--worksheet", type=Path, required=True)
    apply_p.add_argument("--out", type=Path, default=None)
    apply_p.set_defaults(func=cmd_apply)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
