"""Configuration parameters for the three question-generation tiers (Section 7).

One dataclass per tier, plus a ``GenerationConfig`` that carries what all three
share: where the dataset is written, and the provenance stamped onto every
record.

The tier knobs keep their defaults here rather than in ``config.args`` because
they are curation decisions that belong to the tier — Section 7.2's "at most 8
evidence lines", Section 7.3's "at least 4 sentences" — and a default that lived
on the parser could drift from the tier that has to honour it. ``config.args``
declares each one with ``default=None`` and ``_resolve()`` below lets the
dataclass value win, which is what makes "the flag was omitted" and "the flag
was passed the same value as the default" indistinguishable downstream.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_REVIEWER = "faz1_pilot_script"
DEFAULT_CREATED_AT = "2026-08-01T00:00:00Z"


def _resolve(override: T | None, default: T) -> T:
    """Returns ``override`` unless it is ``None``, in which case the dataclass default wins.

    Args:
        override: Value parsed from the CLI, or ``None`` when the flag was omitted.
        default: The owning dataclass's default for that field.

    Returns:
        The value the run should use.
    """
    return default if override is None else override


@dataclass(frozen=True)
class EasyTierParams:
    """Curation knobs for the easy tier (Section 7.1): count / lookup / presence.

    Attributes:
        min_matches: Count literals with fewer real matches are pruned. Presence
            literals are exempt — zero matches there is a valid "No" example,
            not a thin question.
        max_cited_lines: Cap on ``evidence.refs`` per count/presence question.
    """

    min_matches: int = 3
    max_cited_lines: int = 5


@dataclass(frozen=True)
class MediumTierParams:
    """Curation knobs for the medium tier (Section 7.2): single-event explanation.

    Attributes:
        window_size: Evidence lines per question; Section 7.2 caps this at 8.
        questions_per_dataset: Anchor occurrences drafted per LogHub dataset.
            Picks are spread evenly across the file so their evidence windows
            do not overlap.
    """

    window_size: int = 8
    questions_per_dataset: int = 2


@dataclass(frozen=True)
class HardTierParams:
    """Curation knobs for the hard tier (Section 7.3): multi-event synthesis.

    Attributes:
        min_sentences: Minimum sentences demanded of a hard gold answer. This is
            asked of the drafting model in the prompt; the validator re-checks
            it independently, because a prompt is a request, not a guarantee.
        pairs_per_dataset: Non-overlapping group-sets drafted per hard_groups
            spec. Qualifying groups are ranked largest-first and chunked into
            sets of ``num_groups``, so raising this only ever adds sets drawn
            from entities the corpus already has in sufficient volume — it
            never lowers the per-group evidence bar.
        evidence_lines_per_side: A group's first and last this-many lines are
            cited (all of it, unsplit, when the group has ``<= 2x`` this many
            lines). Citing only a group's first few lines understated a burst
            that actually ran for minutes; citing literally every line of a
            group with hundreds of matches measurably degraded the drafted
            answer (the model lost track of which lines belonged to which
            group). First-N-plus-last-N is the version that both keeps the
            group's true start and end in evidence and drafts a correct
            answer.
    """

    min_sentences: int = 4
    pairs_per_dataset: int = 1
    evidence_lines_per_side: int = 15


@dataclass(frozen=True)
class GenerationConfig:
    """What the ``generate`` command writes, and the provenance it stamps.

    ``created_at`` is a fixed string rather than the wall clock: two runs of the
    same configuration against the same pinned corpus must produce byte-identical
    output (Section 6 determinism), and a timestamp is the one field that would
    otherwise differ on every run.

    Attributes:
        out: Dataset file the merged records are written to.
        review_dir: Directory the hard tier writes per-question groundedness
            reports to.
        full: Write the pass to its own scratch file instead of the pilot
            dataset; the generated content is identical either way.
        reviewer: Value written into every record's ``reviewers[]``.
        created_at: Fixed ``gold_provenance.created_at`` timestamp.
        easy: Easy-tier knobs.
        medium: Medium-tier knobs.
        hard: Hard-tier knobs.
    """

    out: Path
    review_dir: Path
    full: bool = False
    reviewer: str = DEFAULT_REVIEWER
    created_at: str = DEFAULT_CREATED_AT
    easy: EasyTierParams = EasyTierParams()
    medium: MediumTierParams = MediumTierParams()
    hard: HardTierParams = HardTierParams()


def get_easy_tier_params(args: Any) -> EasyTierParams:
    """Constructs EasyTierParams from parsed command-line arguments."""
    return EasyTierParams(
        min_matches=_resolve(args.min_matches, EasyTierParams.min_matches),
        max_cited_lines=_resolve(args.max_cited_lines, EasyTierParams.max_cited_lines),
    )


def get_medium_tier_params(args: Any) -> MediumTierParams:
    """Constructs MediumTierParams from parsed command-line arguments."""
    return MediumTierParams(
        window_size=_resolve(args.window_size, MediumTierParams.window_size),
        questions_per_dataset=_resolve(
            args.questions_per_dataset, MediumTierParams.questions_per_dataset
        ),
    )


def get_hard_tier_params(args: Any) -> HardTierParams:
    """Constructs HardTierParams from parsed command-line arguments."""
    return HardTierParams(
        min_sentences=_resolve(args.min_sentences, HardTierParams.min_sentences),
        pairs_per_dataset=_resolve(
            args.hard_pairs_per_dataset, HardTierParams.pairs_per_dataset
        ),
        evidence_lines_per_side=_resolve(
            args.hard_evidence_per_side, HardTierParams.evidence_lines_per_side
        ),
    )


def get_generation_params(args: Any) -> GenerationConfig:
    """Constructs a GenerationConfig, with each tier's knobs nested inside it.

    Args:
        args: Parsed argument namespace containing the output paths, ``--full``,
            the provenance fields and every tier knob.

    Returns:
        GenerationConfig populated from args.
    """
    return GenerationConfig(
        out=args.dataset,
        review_dir=args.review_dir,
        full=args.full,
        reviewer=_resolve(args.reviewer, DEFAULT_REVIEWER),
        created_at=_resolve(args.created_at, DEFAULT_CREATED_AT),
        easy=get_easy_tier_params(args),
        medium=get_medium_tier_params(args),
        hard=get_hard_tier_params(args),
    )
