"""Configuration parameters for the remote Ollama server (Section 5.5).

The two model names are separate fields, never one parameterised "model", and
``src.utils.helper_ollama`` exposes them through two distinct methods
(``draft`` / ``groundedness_check``). Section 5.5/6 requires the model that
writes a gold answer and the model that certifies it to be different families;
keeping them as two fields with two call sites means collapsing the roles takes
a deliberate edit in three places rather than one careless argument.

``config.args._validate_model_separation`` refuses the equal pair at parse time
and ``OllamaClient`` refuses it again at construction, because this config can
also be built directly in a future batch driver that never goes through the CLI.
"""

from dataclasses import dataclass
from typing import Any

from src.params.scale_params import ScaleConfig

DEFAULT_GOLD_DRAFT_MODEL = "nemotron-3-nano:30b"
DEFAULT_GROUNDEDNESS_MODEL = "gpt-oss:20b"


@dataclass(frozen=True)
class OllamaConfig:
    """Connection, model roles and retry policy for the remote Ollama server.

    Attributes:
        base_url: Server address. The server itself is user-provided and
            pre-existing; its operation is not this project's responsibility.
        gold_draft_model: Model that drafts medium/hard gold answers.
        groundedness_model: Model that runs the claim-by-claim groundedness
            check. Must be a different family from ``gold_draft_model``.
        max_parallel_calls: Semaphore width over the whole client (Section 3.2).
        max_retries: Attempts per call before giving up.
        backoff_base_seconds: Base of the exponential retry backoff.
        sql_model: Model that writes the verification SQL for ``verify-answers``.
            Defaults to ``groundedness_model``, which is already a different family
            from the drafting model, so the query that checks an answer never comes
            from the model that wrote it.
        temperature: Sampling temperature; 0.0 keeps drafts reproducible.
        seed: Sampling seed sent with every call. Temperature 0.0 alone is
            greedy decoding, which most servers make deterministic in practice
            but none guarantee across versions; pinning the seed removes the
            remaining freedom, and Section 6's determinism claim stays scoped to
            "same server, same weights" either way.
        num_predict: Token cap per completion. Bounds a runaway draft — an
            uncapped hard-tier answer has no natural stopping point — and makes
            the cost of a scaled pass calculable in advance (Section 3.2).
        num_ctx: Context window requested per call. Left unset, Ollama falls
            back to its own default (typically 2-4k tokens) regardless of what
            the model itself supports, which silently truncates a long prompt
            rather than erroring — the hard tier's evidence blocks can run to
            tens of thousands of tokens now that a group's lines are cited in
            full, so this must be requested explicitly rather than assumed.
        require_models: Names ``check-ollama`` demands from the server. ``None``
            means "the two role models above", which is the honest default —
            those are exactly the models a full run would need.
    """

    base_url: str
    gold_draft_model: str = DEFAULT_GOLD_DRAFT_MODEL
    groundedness_model: str = DEFAULT_GROUNDEDNESS_MODEL
    sql_model: str = DEFAULT_GROUNDEDNESS_MODEL
    max_parallel_calls: int = 4
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    temperature: float = 0.0
    seed: int = 7
    num_predict: int = 512
    num_ctx: int = 32768
    require_models: tuple[str, ...] | None = None

    @property
    def role_models(self) -> tuple[tuple[str, str], ...]:
        """Returns the ``(model_name, role)`` pairs a full generation run uses."""
        roles = [
            (self.gold_draft_model, "gold_draft"),
            (self.groundedness_model, "groundedness_check"),
        ]
        if self.sql_model not in (self.gold_draft_model, self.groundedness_model):
            roles.append((self.sql_model, "sql_verification"))
        return tuple(roles)

    @property
    def required_model_names(self) -> tuple[str, ...]:
        """Returns the model names ``check-ollama`` must find on the server."""
        if self.require_models:
            return tuple(self.require_models)
        return tuple(name for name, _role in self.role_models)


def get_ollama_params(
    args: Any, scale_config: ScaleConfig | None = None
) -> OllamaConfig:
    """Constructs an OllamaConfig from parsed args, with scale_config as fallback.

    ``max_parallel_calls`` has two possible sources and a defined precedence:
    an explicit ``--max_parallel_model_calls`` wins, otherwise the
    ``ScaleConfig``'s value (the scaling knob of Section 3.2), otherwise the
    dataclass default. ``main.py`` passes the scale config for every command,
    so the semaphore a generate pass runs under is always the one the scaling
    configuration names.

    Args:
        args: Parsed argument namespace.
        scale_config: The run's ``ScaleConfig``, or ``None`` when a caller has
            no scaling context (direct library use).

    Returns:
        OllamaConfig populated from args and, where relevant, scale_config.
    """
    if args.max_parallel_model_calls is not None:
        max_parallel = args.max_parallel_model_calls
    elif scale_config is not None:
        max_parallel = scale_config.max_parallel_model_calls
    else:
        max_parallel = OllamaConfig.max_parallel_calls

    return OllamaConfig(
        base_url=args.base_url,
        gold_draft_model=args.gold_draft_model,
        groundedness_model=args.groundedness_model,
        sql_model=(
            args.groundedness_model if args.sql_model is None else args.sql_model
        ),
        max_parallel_calls=max_parallel,
        max_retries=(
            OllamaConfig.max_retries if args.max_retries is None else args.max_retries
        ),
        backoff_base_seconds=(
            OllamaConfig.backoff_base_seconds
            if args.backoff_base_seconds is None
            else args.backoff_base_seconds
        ),
        temperature=(
            OllamaConfig.temperature if args.temperature is None else args.temperature
        ),
        seed=(OllamaConfig.seed if args.seed is None else args.seed),
        num_predict=(
            OllamaConfig.num_predict if args.num_predict is None else args.num_predict
        ),
        num_ctx=(OllamaConfig.num_ctx if args.num_ctx is None else args.num_ctx),
        require_models=tuple(args.require_models) if args.require_models else None,
    )
