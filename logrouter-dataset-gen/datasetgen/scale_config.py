#!/usr/bin/env python3
"""
scale_config.py

Typed view of the scale_config.yaml fields main.py's `generate` command
actually reads, structured to mirror the YAML file's own top-level sections
(run: / difficulty_mix: / concurrency:) so it is unambiguous which section a
given parameter comes from -- instead of a chain of config.get("x", {}).get(
"y", default) calls inline in main.py.

Fields declared in scale_config.yaml but not read by the current pipeline
(datasets[].weight, phrasing_families_min_per_intent, models,
concurrency.retry_max_attempts/retry_backoff_base_seconds) are intentionally
left unmodeled here rather than silently reintroduced.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    """scale_config.yaml: run:"""
    target_total_questions: int = 100


@dataclass(frozen=True)
class DifficultyMix:
    """scale_config.yaml: difficulty_mix:"""
    easy: float = 0.70
    medium: float = 0.20
    hard: float = 0.10


@dataclass(frozen=True)
class ConcurrencyConfig:
    """scale_config.yaml: concurrency:"""
    max_parallel_model_calls: int = 4


@dataclass(frozen=True)
class ScaleConfig:
    run: RunConfig
    difficulty_mix: DifficultyMix
    concurrency: ConcurrencyConfig

    @classmethod
    def load(cls, path: Path) -> "ScaleConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        run_raw = raw.get("run") or {}
        mix_raw = raw.get("difficulty_mix") or {}
        concurrency_raw = raw.get("concurrency") or {}
        return cls(
            run=RunConfig(
                target_total_questions=run_raw.get(
                    "target_total_questions", RunConfig.target_total_questions),
            ),
            difficulty_mix=DifficultyMix(
                easy=mix_raw.get("easy", DifficultyMix.easy),
                medium=mix_raw.get("medium", DifficultyMix.medium),
                hard=mix_raw.get("hard", DifficultyMix.hard),
            ),
            concurrency=ConcurrencyConfig(
                max_parallel_model_calls=concurrency_raw.get(
                    "max_parallel_model_calls", ConcurrencyConfig.max_parallel_model_calls),
            ),
        )
