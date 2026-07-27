"""Scaling parameters: how large a run is, and how fast it may call the model.

These used to live in a ``scale_config.yaml`` read at startup. They are
dataclasses now, for the reason Section 3.2 wanted a config file in the first
place: scaling a run from 100 to 3000 questions must not require a code change.
A CLI flag backed by a dataclass default satisfies that better than a YAML file
did — ``--target_total_questions 3000`` is one argument, whereas the file had to
be edited, mounted into the container, and kept in sync with the dataclasses that
already modelled its three fields.

So the single configuration source is ``config/args.py`` plus these dataclasses,
and the project keeps exactly one non-Python config artifact:
``config/question_schema.json``, which exists because ``jsonschema`` has to read
a schema document.

``difficulty_mix`` is a reporting target, not a quota. Nothing enforces it: the
tiers produce what the curated specs and the corpus allow, and a pilot pass that
lands off the mix is a fact for the summary line rather than an error (Section
3.2). Enforcing it would mean discarding valid questions to hit a ratio.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DifficultyMix:
    """Target share per difficulty tier, for reporting.

    Attributes:
        easy: Target share of easy-tier questions.
        medium: Target share of medium-tier questions.
        hard: Target share of hard-tier questions.
    """

    easy: float = 0.70
    medium: float = 0.20
    hard: float = 0.10


@dataclass(frozen=True)
class ScaleConfig:
    """How large a run is meant to be, and its model-call concurrency.

    Attributes:
        target_total_questions: The pass's question target. Reported beside the
            realised counts, never enforced.
        difficulty_mix: Target share per tier.
        max_parallel_model_calls: Semaphore width over the Ollama client.
    """

    target_total_questions: int = 100
    difficulty_mix: DifficultyMix = DifficultyMix()
    max_parallel_model_calls: int = 4


def get_scale_params(args: Any) -> ScaleConfig:
    """Constructs a ScaleConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace carrying ``target_total_questions`` and
            ``max_parallel_model_calls``.

    Returns:
        ScaleConfig populated from args, with the dataclass default winning
        wherever the flag was omitted.
    """
    return ScaleConfig(
        target_total_questions=(
            ScaleConfig.target_total_questions
            if args.target_total_questions is None
            else args.target_total_questions
        ),
        max_parallel_model_calls=(
            ScaleConfig.max_parallel_model_calls
            if args.max_parallel_model_calls is None
            else args.max_parallel_model_calls
        ),
    )
