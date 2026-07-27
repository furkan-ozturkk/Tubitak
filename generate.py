"""Question generation (Section 7).

A library, not a script: ``main.py`` is the only executable entry point, and
``run_generation`` is what its ``generate`` command calls. Each tier function loops
the curated datasets, hands each ``corpus_provider()`` view to the matching builder
in ``src.generators``, and returns records.

Two passes exist, and the narrow one is the default:

``generate_official_set()`` is the official stage-1 output — 1 count + 1 presence
question per LogHub dataset, no model involved, every answer reproducible by SQL.

``generate_full()`` runs all three tiers and needs Ollama. Its medium and hard
records leave here ``review_status=in_review`` and it writes to its own file, so a
draft pass never lands on top of the verified stage-1 output. They join the official
output only after a human has reviewed them (``main.py --command review-export`` /
``review-apply``), which is what grows the 20 upward under the staged scaling plan
of Section 3.

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
    select_official_20,
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
    corpus_config: CorpusConfig, config: GenerationConfig
) -> tuple[list[dict[str, Any]], GenerationSummary]:
    """Produces the official 20-question stage-1 set (Section 3.1).

    The full easy tier is built and then narrowed, rather than a narrower tier being
    generated directly. The selection has to pick records the easy tier really
    produced, pruning included, so a literal that turned out to be too thin drops out
    of the official set instead of appearing in it with an answer no other pass would
    have given. The cost is the discarded records, which is a few extra SQL counts
    over a 2k-line corpus.

    Args:
        corpus_config: Where the corpus is read from and how it is partitioned.
        config: Generation config.

    Returns:
        Tuple ``(records, summary)``.
    """
    print("=== Official 20-question set (easy tier only, no model) ===")
    easy_records = generate_easy(corpus_config, config)
    dataset_keys = [name.lower() for name in DATASET_SPECS]
    records = select_official_20(easy_records, dataset_keys)
    summary = GenerationSummary(easy=len(records), out=config.out, official_set=True)
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

    The Ollama client is built only for a ``--full`` pass. The default official set
    is pure SQL and has to stay runnable with no model server in reach, and a client
    constructed for it would fail a run for a reason that has nothing to do with its
    own work.

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

    try:
        if not config.full:
            records, summary = generate_official_set(corpus_config, config)
        else:
            print(f"Gold draft   : {ollama_config.gold_draft_model}")
            print(f"Groundedness : {ollama_config.groundedness_model}")
            print(f"Parallel     : {ollama_config.max_parallel_calls}")
            client = OllamaClient(ollama_config)
            records, summary = generate_full(corpus_config, config, client)
            summary.target_total = scale_config.target_total_questions
            summary.difficulty_mix = scale_config.difficulty_mix

        resolve_splits(records, corpus_config.test_fraction)
        write_json(config.out, records)
        print_generation_summary(summary)
    finally:
        helper_postgres.close_connection()
    return 0
