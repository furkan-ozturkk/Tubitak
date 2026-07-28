"""Question generation (Section 7).

A library, not a script: ``main.py`` is the only executable entry point, and
``run_generation`` is what its ``generate`` command calls. Each tier function loops
the curated datasets, hands each ``corpus_provider()`` view to the matching builder
in ``src.generators``, and returns records.

Two passes exist, and the mixed one is the default:

``generate_official_set()`` is the official stage-1 output — 7 easy + 7 medium + 6
hard, 20 questions total. It needs Ollama, since medium and hard are model-drafted;
the medium/hard records it selects arrive ``review_status=in_review`` exactly as
they would from a ``--full`` pass, and need the same ``review-export`` /
``review-apply`` cycle before they count as verified. Only easy is self-certifying
(re-derivable by SQL); the official set was pure-easy before this mix, which is
why ``select_official_mixed_set`` documents the exact per-dataset assignment.

``generate_full()`` runs all three tiers at full width (every dataset, every
occurrence) rather than the official set's fixed 20, and writes to its own file so a
draft pass never lands on top of the official output.

``_OFFICIAL_EASY_DATASETS`` / ``_OFFICIAL_MEDIUM_DATASETS`` are the fixed,
deterministic dataset assignment the mix reads from: hard picks the 6 datasets
that have a hard_group (one each); easy and medium each pick 7 of the 10
datasets, chosen so every dataset contributes at least one question -- the 4
hard-incapable datasets (apache, windows, mac, openstack) get an easy and/or
medium slot instead of a hard one.

Splits are assigned once, after every record exists, by
``src.utils.helper_splits.resolve_splits``. No tier sets its own: a tier sees only
its own records and therefore cannot know which of its evidence groups a later hard
question will link to another, which is exactly the case where a per-record decision
leaks an event across the dev/test boundary.
"""

from typing import Any

from src.data.data_factory import corpus_provider
from src.data.dataset_specs import DATASET_SPECS
from src.generators import (
    build_easy_records,
    build_hard_records,
    build_medium_records,
    select_official_easy,
)
from src.params.corpus_params import CorpusConfig
from src.params.generation_params import GenerationConfig
from src.params.ollama_params import OllamaConfig
from src.params.results_params import GenerationSummary
from src.params.scale_params import ScaleConfig
from src.utils import helper_postgres
from src.utils.helper_ollama import OllamaClient
from src.utils.helper_run import print_generation_summary, write_json
from src.utils.helper_splits import resolve_splits

_OFFICIAL_EASY_DATASETS = [
    "linux",
    "apache",
    "windows",
    "mac",
    "hdfs",
    "openssh",
    "bgl",
]
_OFFICIAL_MEDIUM_DATASETS = [
    "hadoop",
    "zookeeper",
    "openstack",
    "linux",
    "apache",
    "windows",
    "mac",
]


def select_official_mixed_set(
    easy_records: list[dict[str, Any]],
    medium_records: list[dict[str, Any]],
    hard_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Narrows the three tiers to the official 7 easy + 7 medium + 6 hard set.

    Every LogHub dataset contributes at least one question; the four datasets
    without a hard_group (apache, windows, mac, openstack) contribute to easy
    and/or medium instead, per ``_OFFICIAL_EASY_DATASETS`` /
    ``_OFFICIAL_MEDIUM_DATASETS`` above. Hard takes every hard-tier record
    produced, since exactly one hard_group is curated per hard-capable dataset
    (Section 7.3) -- there is nothing to narrow.

    Args:
        easy_records: Every easy-tier record produced this pass.
        medium_records: Every medium-tier record produced this pass.
        hard_records: Every hard-tier record produced this pass.

    Returns:
        20 records: 7 easy (first-phrasing count) + 7 medium (first
        occurrence) + 6 hard (one per hard-capable dataset).
    """
    selected_easy = select_official_easy(easy_records, _OFFICIAL_EASY_DATASETS)

    medium_by_id = {r["id"]: r for r in medium_records}
    selected_medium = []
    for key in _OFFICIAL_MEDIUM_DATASETS:
        match = next(
            (
                rid
                for rid in medium_by_id
                if rid.startswith(f"{key}_v1_semantic_") and rid.endswith("_0")
            ),
            None,
        )
        if match:
            selected_medium.append(medium_by_id[match])

    return selected_easy + selected_medium + list(hard_records)


def generate_easy(
    corpus_config: CorpusConfig, config: GenerationConfig
) -> list[dict[str, Any]]:
    """Runs the easy tier across every curated dataset.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config.

    Returns:
        Every easy-tier record, without ``split``.
    """
    all_records: list[dict[str, Any]] = []
    for name, spec in DATASET_SPECS.items():
        view = corpus_provider(corpus_config, spec)
        records = build_easy_records(view, spec, config)
        print(f"[{name}] easy: {len(records)} questions")
        all_records.extend(records)
    return all_records


def generate_medium(
    corpus_config: CorpusConfig, config: GenerationConfig, client: OllamaClient
) -> list[dict[str, Any]]:
    """Runs the medium tier across every curated dataset.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config.
        client: Ollama client used for the gold drafts.

    Returns:
        Every medium-tier record, without ``split``, all ``in_review``.
    """
    all_records: list[dict[str, Any]] = []
    for name, spec in DATASET_SPECS.items():
        view = corpus_provider(corpus_config, spec)
        records = build_medium_records(view, spec, config, client)
        print(f"[{name}] medium: {len(records)} questions")
        all_records.extend(records)
    return all_records


def generate_hard(
    corpus_config: CorpusConfig, config: GenerationConfig, client: OllamaClient
) -> list[dict[str, Any]]:
    """Runs the hard tier across every curated dataset.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config; ``review_dir`` receives the groundedness reports.
        client: Ollama client used for the drafts and the groundedness checks.

    Returns:
        Every hard-tier record, without ``split``, all ``in_review``.
    """
    all_records: list[dict[str, Any]] = []
    for name, spec in DATASET_SPECS.items():
        view = corpus_provider(corpus_config, spec)
        records = build_hard_records(view, spec, config, client)
        print(f"[{name}] hard: {len(records)} questions")
        all_records.extend(records)
    return all_records


def generate_official_set(
    corpus_config: CorpusConfig, config: GenerationConfig, client: OllamaClient
) -> tuple[list[dict[str, Any]], GenerationSummary]:
    """Produces the official 7 easy + 7 medium + 6 hard, 20-question set (Section 3.1).

    Every tier is built at full width and then narrowed, rather than a narrower
    pass being generated directly. The selection has to pick records each tier
    really produced, pruning included, so a literal or anchor that turned out to
    be too thin drops out of the official set instead of appearing in it with an
    answer no other pass would have given.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config.
        client: Ollama client used by the medium and hard tiers.

    Returns:
        Tuple ``(records, summary)``.
    """
    print("=== Official 20-question set (7 easy + 7 medium + 6 hard) ===")
    easy_records = generate_easy(corpus_config, config)
    medium_records = generate_medium(corpus_config, config, client)
    hard_records = generate_hard(corpus_config, config, client)
    records = select_official_mixed_set(easy_records, medium_records, hard_records)
    summary = GenerationSummary(
        easy=sum(1 for r in records if r["difficulty"] == "easy"),
        medium=sum(1 for r in records if r["difficulty"] == "medium"),
        hard=sum(1 for r in records if r["difficulty"] == "hard"),
        out=config.out,
        official_set=True,
    )
    return records, summary


def generate_full(
    corpus_config: CorpusConfig, config: GenerationConfig, client: OllamaClient
) -> tuple[list[dict[str, Any]], GenerationSummary]:
    """Runs all three tiers and returns their merged records.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config.
        client: Ollama client used by the medium and hard tiers.

    Returns:
        Tuple ``(records, summary)``.
    """
    print("=== Easy (deterministic) ===")
    easy_records = generate_easy(corpus_config, config)

    print("\n=== Medium (semantic) ===")
    medium_records = generate_medium(corpus_config, config, client)

    print("\n=== Hard ===")
    hard_records = generate_hard(corpus_config, config, client)

    summary = GenerationSummary(
        easy=len(easy_records),
        medium=len(medium_records),
        hard=len(hard_records),
        out=config.out,
    )
    return easy_records + medium_records + hard_records, summary


def run_generation(
    corpus_config: CorpusConfig,
    config: GenerationConfig,
    scale_config: ScaleConfig,
    ollama_config: OllamaConfig,
) -> int:
    """Runs one generate pass end to end and writes the dataset.

    The Ollama client is now built for both passes: the official set mixes in
    medium and hard tiers (Section 3.1), so it needs a model server exactly as
    much as ``--full`` does. Only ``validate.py``'s SQL-based checks remain
    model-free.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config, including the output path.
        scale_config: Reporting target and model-call concurrency.
        ollama_config: Server address and the two role models.

    Returns:
        ``0``.
    """
    print(f"Command      : generate{' --full' if config.full else ''}")
    print(f"Corpus dir   : {corpus_config.corpus_dir}")
    print(f"Output       : {config.out}")
    print(f"Test fraction: {corpus_config.test_fraction}")
    print(f"Gold draft   : {ollama_config.gold_draft_model}")
    print(f"Groundedness : {ollama_config.groundedness_model}")
    print(f"Parallel     : {ollama_config.max_parallel_calls}")

    try:
        client = OllamaClient(ollama_config)
        if not config.full:
            records, summary = generate_official_set(corpus_config, config, client)
        else:
            records, summary = generate_full(corpus_config, config, client)
            summary.target_total = scale_config.target_total_questions
            summary.difficulty_mix = scale_config.difficulty_mix

        resolve_splits(records, corpus_config.test_fraction)
        write_json(config.out, records)
        print_generation_summary(summary)
    finally:
        helper_postgres.close_connection()
    return 0
