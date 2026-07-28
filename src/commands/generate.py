"""Question generation (Section 7).

A library, not a script: ``main.py`` is the only executable entry point, and
``run_generation`` is what its ``generate`` command calls. Each tier function loops
the curated datasets, hands each ``corpus_provider()`` view to the matching builder
in ``src.generators``, and returns records.

Every pass runs all three tiers at full width — every dataset, every phrasing
family, every anchor occurrence, every hard group the specs can fill. An earlier
revision narrowed the default pass to a 20-question set (one count phrasing per
dataset plus a medium/hard sample), and that narrowing is exactly what the
acceptance criteria forbid shipping: Section 2/7.4 requires a deterministic
intent to be reachable through at least three phrasing families, and a router
with a keyword path cannot be evaluated by a set holding zero lookup questions.
The narrowed set survives only in ``output/archive/``.

``--full`` no longer changes what is generated — it changes only *where* the
output lands (``config.args._resolve_paths`` moves it off the pilot file), so an
experimental pass can never overwrite a pilot dataset whose medium/hard records
a human has since reviewed. Medium- and hard-tier records leave every pass as
``review_status=in_review`` and need the ``review-export`` / ``review-apply``
cycle before they count as verified; only the easy tier is self-certifying
(re-derivable by SQL).

Splits are assigned once, after every record exists, by
``src.utils.helper_splits.resolve_splits``. No tier sets its own: a tier sees only
its own records and therefore cannot know which components its evidence belongs
to — a hard question links groups by co-citation, and any two questions that cite
the same corpus line are linked through that line, both of which are exactly the
cases where a per-record decision leaks an event across the dev/test boundary.
"""

from typing import Any

from src.data.data_factory import corpus_provider
from src.data.dataset_specs import DATASET_SPECS
from src.generators import (
    build_easy_records,
    build_hard_records,
    build_medium_records,
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


def generate_all_tiers(
    corpus_config: CorpusConfig, config: GenerationConfig, client: OllamaClient
) -> tuple[list[dict[str, Any]], GenerationSummary]:
    """Runs all three tiers at full width and returns their merged records.

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
        official_set=not config.full,
    )
    return easy_records + medium_records + hard_records, summary


def run_generation(
    corpus_config: CorpusConfig,
    config: GenerationConfig,
    scale_config: ScaleConfig,
    ollama_config: OllamaConfig,
) -> int:
    """Runs one generate pass end to end and writes the dataset.

    The Ollama client is built for every pass: medium and hard gold is
    model-drafted, so generation needs a model server. Only ``validate.py``'s
    SQL-based checks remain model-free.

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
        records, summary = generate_all_tiers(corpus_config, config, client)
        summary.target_total = scale_config.target_total_questions
        summary.difficulty_mix = scale_config.difficulty_mix

        resolve_splits(records, corpus_config.test_fraction)
        write_json(config.out, records)
        print_generation_summary(summary)
    finally:
        helper_postgres.close_connection()
    return 0
