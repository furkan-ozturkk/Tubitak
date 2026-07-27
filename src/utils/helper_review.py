"""Human review of model-drafted gold answers (Section 7.3 step 5 / Section 6).

Every ``review_status=in_review`` record — all medium- and hard-tier gold — needs a
human accept/edit/reject decision before it can become ``verified``. Nothing here
makes that decision: ``export_worksheet`` writes a worksheet for a human to fill in
and ``apply_worksheet`` writes their decisions back. A row with no decision, or an
``edit`` with no replacement text, is left ``in_review`` rather than resolved by
default, because the whole point of the step is that a human said so.

Three properties make a decision auditable rather than merely recorded.

**The worksheet is bound to what it was exported from.** Each row carries the
digest of the draft answer it was written against, and ``apply_worksheet`` refuses a
row whose draft has since changed. Without that, a decision made on an old draft
would silently certify a new one — the realistic failure being a regenerated
dataset and a worksheet from the previous pass.

**The decision names who made it and when.** ``--reviewer`` is appended to the
record's ``reviewers`` list and a review event carrying the reviewer, the timestamp
and the input digests is appended to a log beside the dataset. The record's own
``reviewers`` field previously held only the generating script, which meant a
``verified`` record asserted human review without naming a human.

**A worksheet cannot decide the same record twice.** Duplicate ids are rejected
rather than resolved last-writer-wins, which is what a hand-edited CSV produces
when rows are copied.

CSV fields are escaped against spreadsheet formula injection. A log line beginning
with ``=`` or ``+`` is ordinary in this corpus and would otherwise be evaluated when
the reviewer opens the worksheet.
"""

import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from src.params.review_params import ReviewApplyConfig, ReviewExportConfig
from src.utils.helper_run import write_json, write_text

WORKSHEET_FIELDS = [
    "id",
    "difficulty",
    "question",
    "draft_answer",
    "draft_sha256",
    "groundedness_summary",
    "evidence",
    "decision",
    "edited_answer",
    "note",
]

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _sha256_text(text: str) -> str:
    """Returns the ``sha256:``-prefixed digest of a string.

    Args:
        text: Text to hash.

    Returns:
        The prefixed hex digest.
    """
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _csv_safe(value: str) -> str:
    """Neutralises a value that a spreadsheet would read as a formula.

    Args:
        value: Cell value.

    Returns:
        The value, prefixed with an apostrophe when it starts with a character
        that begins a formula.
    """
    if value and value[0] in FORMULA_PREFIXES:
        return "'" + value
    return value


def _groundedness_summary(review_dir: Path | None, question_id: str) -> str:
    """Summarises one question's groundedness report for the worksheet.

    Args:
        review_dir: Directory of per-question groundedness reports, or ``None``.
        question_id: The record id whose report to summarise.

    Returns:
        A ``"<supported>/<total> claims supported"`` string, or ``""`` when there is
        no report — the normal case for medium-tier records, since only the hard
        tier runs a per-claim check.
    """
    if not review_dir:
        return ""
    report_path = Path(review_dir) / f"{question_id}.json"
    if not report_path.exists():
        return ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    supported = [claim["supported"] for claim in report.get("claims", [])]
    if not supported:
        return ""
    return f"{supported.count('yes')}/{len(supported)} claims supported"


def _evidence_text(record: dict[str, Any]) -> str:
    """Renders a record's cited evidence for the worksheet.

    The reviewer is being asked whether an answer is grounded in the evidence, so
    the evidence has to be in front of them. A summary count alone asks them to
    trust the model that produced the count.

    Args:
        record: A question record.

    Returns:
        One ``dataset:line`` reference per line, newline-separated.
    """
    refs = record.get("evidence", {}).get("refs", [])
    return "\n".join(
        f"{ref.get('group_id', '?')} #{ref.get('line_number', '?')}" for ref in refs
    )


def export_worksheet(config: ReviewExportConfig) -> int:
    """Exports every in_review record to a CSV worksheet.

    Args:
        config: Dataset, worksheet destination and groundedness report directory.

    Returns:
        ``0``.
    """
    questions = json.loads(Path(config.dataset).read_text(encoding="utf-8"))
    rows = []
    for question in questions:
        if question.get("review_status") != "in_review":
            continue
        draft_answer = question["expected_answer"]
        rows.append(
            {
                "id": question["id"],
                "difficulty": question["difficulty"],
                "question": _csv_safe(question["question"]),
                "draft_answer": _csv_safe(draft_answer),
                "draft_sha256": _sha256_text(draft_answer),
                "groundedness_summary": _groundedness_summary(
                    config.review_dir, question["id"]
                ),
                "evidence": _csv_safe(_evidence_text(question)),
                "decision": "",
                "edited_answer": "",
                "note": "",
            }
        )

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=WORKSHEET_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(Path(config.worksheet), buffer.getvalue())

    print(f"Exported {len(rows)} in_review record(s) to {config.worksheet}")
    if not rows:
        print(
            "  Note: the dataset holds no in_review records. The official "
            "20-question set is entirely deterministic and verified; point "
            "--dataset at a --full output to review model-drafted gold."
        )
    return 0


def _validate_worksheet(
    decisions: list[dict[str, str]], by_id: dict[str, dict[str, Any]]
) -> list[str]:
    """Checks a filled-in worksheet against the dataset before applying anything.

    Applying is all-or-nothing: a worksheet with any structural problem is rejected
    before the first record changes, so a half-applied review cannot leave the
    dataset in a state nobody chose.

    Args:
        decisions: Rows read from the worksheet.
        by_id: Dataset records, keyed by id.

    Returns:
        Human-readable problems; empty when the worksheet is safe to apply.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(decisions, start=2):
        question_id = (row.get("id") or "").strip()
        if not question_id:
            problems.append(f"row {row_number}: empty id")
            continue
        if question_id in seen:
            problems.append(
                f"row {row_number}: duplicate decision for id {question_id}; "
                f"a worksheet must decide each record once"
            )
            continue
        seen.add(question_id)

        record = by_id.get(question_id)
        if record is None:
            problems.append(
                f"row {row_number}: id {question_id} is not in the dataset"
            )
            continue

        claimed_digest = (row.get("draft_sha256") or "").strip()
        actual_digest = _sha256_text(record["expected_answer"])
        if claimed_digest and claimed_digest != actual_digest:
            problems.append(
                f"row {row_number}: id {question_id} was reviewed against a draft "
                f"that has since changed (worksheet {claimed_digest[:19]}, dataset "
                f"{actual_digest[:19]}); re-export the worksheet"
            )

        decision = (row.get("decision") or "").strip().lower()
        if decision and decision not in ("accept", "edit", "reject"):
            problems.append(
                f"row {row_number}: unknown decision '{decision}'; "
                f"expected accept, edit or reject"
            )
    return problems


def apply_worksheet(config: ReviewApplyConfig) -> int:
    """Applies a filled-in worksheet back onto the dataset.

    Four outcomes, and the two that change nothing are reported as loudly as the two
    that do: a row whose decision is missing or whose edit is empty leaves its
    record ``in_review``. The printed tally is what tells a reviewer whether the file
    they edited is the file that was applied.

    Args:
        config: Dataset, filled-in worksheet, output path and reviewer identity.

    Returns:
        ``0`` applied, ``1`` the worksheet was rejected and nothing was written.
    """
    dataset_path = Path(config.dataset)
    questions = json.loads(dataset_path.read_text(encoding="utf-8"))
    by_id = {question["id"]: question for question in questions}

    worksheet_bytes = Path(config.worksheet).read_bytes()
    with open(config.worksheet, "r", encoding="utf-8", newline="") as handle:
        decisions = list(csv.DictReader(handle))

    problems = _validate_worksheet(decisions, by_id)
    if problems:
        print(f"REJECTED: worksheet has {len(problems)} problem(s); nothing was written.")
        for problem in problems[:20]:
            print(f"  - {problem}")
        if len(problems) > 20:
            print(f"  ... (+{len(problems) - 20} more)")
        return 1

    timestamp = (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    events: list[dict[str, Any]] = []
    accepted = edited = rejected = skipped = 0

    for row in decisions:
        question_id = row["id"].strip()
        record = by_id[question_id]
        decision = (row.get("decision") or "").strip().lower()
        note = (row.get("note") or "").strip()

        if decision == "accept":
            record["review_status"] = "verified"
            accepted += 1
        elif decision == "edit":
            edited_answer = (row.get("edited_answer") or "").strip()
            if not edited_answer:
                skipped += 1
                continue
            record["expected_answer"] = edited_answer
            record["review_status"] = "verified"
            edited += 1
        elif decision == "reject":
            record["review_status"] = "rejected"
            rejected += 1
        else:
            skipped += 1
            continue

        if config.reviewer not in record["reviewers"]:
            record["reviewers"].append(config.reviewer)

        events.append(
            {
                "question_id": question_id,
                "decision": decision,
                "reviewer": config.reviewer,
                "reviewed_at": timestamp,
                "draft_sha256": (row.get("draft_sha256") or "").strip() or None,
                "final_sha256": _sha256_text(record["expected_answer"]),
                "worksheet_sha256": "sha256:"
                + hashlib.sha256(worksheet_bytes).hexdigest(),
                "note": note or None,
            }
        )

    write_json(Path(config.out), questions)
    if events:
        existing: list[dict[str, Any]] = []
        if Path(config.event_log).exists():
            existing = json.loads(Path(config.event_log).read_text(encoding="utf-8"))
        write_json(Path(config.event_log), existing + events)

    print(
        f"Applied worksheet: accepted={accepted} edited={edited} rejected={rejected} "
        f"left_undecided={skipped}"
    )
    print(f"Reviewer: {config.reviewer} at {timestamp}")
    print(f"Wrote updated dataset to {config.out}")
    if events:
        print(f"Appended {len(events)} review event(s) to {config.event_log}")
    return 0
