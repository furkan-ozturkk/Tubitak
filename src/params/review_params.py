"""Configuration parameters for the human-review commands (Section 7.3 step 5).

Export and apply are two dataclasses rather than one, because they disagree about
the fields that matter: apply writes, export does not, and only apply needs a
reviewer identity. Giving export an ``out`` it never uses would make it look, at the
call site, like a command that can modify the dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REVIEWER = "unattributed-human-review"


@dataclass(frozen=True)
class ReviewExportConfig:
    """``review-export``: in_review records out to a CSV worksheet.

    Attributes:
        dataset: Question dataset to export ``review_status=in_review`` records from.
        worksheet: CSV worksheet written for a human to fill in.
        review_dir: Directory of per-question groundedness reports, summarised into
            the worksheet's ``groundedness_summary`` column.
    """

    dataset: Path
    worksheet: Path
    review_dir: Path | None = None


@dataclass(frozen=True)
class ReviewApplyConfig:
    """``review-apply``: a filled-in worksheet back onto the dataset.

    ``reviewer`` defaults to a name that reads as an admission rather than as a
    person. A ``verified`` record asserts that a human accepted a model's draft, so
    the record has to say which human; an anonymous default that looked like an
    identity would let that assertion be made by nobody.

    Attributes:
        dataset: Question dataset the decisions are applied to.
        worksheet: Filled-in CSV worksheet.
        out: Where the updated dataset is written. Resolved by
            ``config.args._resolve_paths`` to ``dataset`` itself when the flag was
            omitted, so this is never ``None`` in practice.
        reviewer: Identity appended to each decided record's ``reviewers`` list and
            recorded in every review event.
        event_log: Append-only JSON log of review decisions.
        review_dir: Directory of per-question groundedness reports. Apply reads
            them to refuse an ``accept`` on a record whose draft holds a claim
            the groundedness model marked unsupported — such a record needs an
            ``edit`` or a ``reject``, not a pass-through.
    """

    dataset: Path
    worksheet: Path
    out: Path
    reviewer: str = DEFAULT_REVIEWER
    event_log: Path = Path("/output/pilot/review/review_events.json")
    review_dir: Path | None = None


def get_review_export_params(args: Any) -> ReviewExportConfig:
    """Constructs a ReviewExportConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        ReviewExportConfig populated from args.
    """
    return ReviewExportConfig(
        dataset=args.dataset,
        worksheet=args.worksheet,
        review_dir=args.review_dir,
    )


def get_review_apply_params(args: Any) -> ReviewApplyConfig:
    """Constructs a ReviewApplyConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        ReviewApplyConfig populated from args, with the dataclass default winning
        wherever the flag was omitted.
    """
    return ReviewApplyConfig(
        dataset=args.dataset,
        worksheet=args.worksheet,
        out=args.review_out,
        reviewer=DEFAULT_REVIEWER if args.reviewer is None else args.reviewer,
        event_log=(
            ReviewApplyConfig.event_log if args.review_log is None else args.review_log
        ),
        review_dir=args.review_dir,
    )
