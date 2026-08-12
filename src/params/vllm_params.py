"""Configuration parameters for the local vLLM servers (Section 5.5).

The gold-draft and groundedness-check roles now point at two independently
addressed vLLM servers rather than one Ollama server hosting a shared model
catalogue: each vLLM process serves exactly one model, so the connection
address is itself a per-role field alongside the model id. Splitting the old
single ``base_url`` into ``gold_draft_base_url`` and ``groundedness_base_url``
keeps the Scientific Integrity Rule enforceable the same way it always was —
the two roles are two fields with two call sites, and collapsing them onto the
same server and model still takes a deliberate edit in three places rather
than one careless argument.

``config.args._validate_model_separation`` refuses the equal-model pair at
parse time and ``VllmClient`` refuses it again at construction, because this
config can also be built directly in a future batch driver that never goes
through the CLI.
"""

from dataclasses import dataclass
from typing import Any

from src.params.scale_params import ScaleConfig

DEFAULT_GOLD_DRAFT_BASE_URL = "http://furkan.ozturk_vllm:8003"
DEFAULT_GOLD_DRAFT_MODEL = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
DEFAULT_GROUNDEDNESS_BASE_URL = "http://furkan.ozturk_vllm:8001"
DEFAULT_GROUNDEDNESS_MODEL = "Qwen/Qwen3-32B-AWQ"


@dataclass(frozen=True)
class VllmConfig:
    """Connection, model roles and retry policy for the two role servers.

    Attributes:
        gold_draft_base_url: vLLM server that drafts medium/hard gold answers.
        gold_draft_model: Model id that server was started with.
        groundedness_base_url: vLLM server that runs the claim-by-claim
            groundedness check. Must be a different family from
            ``gold_draft_model``.
        groundedness_model: Model id that server was started with.
        sql_base_url: vLLM server that writes the verification SQL for
            ``verify-answers``. Defaults to ``groundedness_base_url``, which is
            already a different family from the drafting model, so the query
            that checks an answer never comes from the model that wrote it.
        sql_model: Model id ``sql_base_url`` was started with. Defaults to
            ``groundedness_model``.
        max_parallel_calls: Semaphore width over the whole client (Section 3.2).
        max_retries: Attempts per call before giving up.
        backoff_base_seconds: Base of the exponential retry backoff.
        temperature: Sampling temperature; 0.0 keeps drafts reproducible.
        seed: Sampling seed sent with every call. Temperature 0.0 alone is
            greedy decoding, which most servers make deterministic in practice
            but none guarantee across versions; pinning the seed removes the
            remaining freedom, and Section 6's determinism claim stays scoped
            to "same server, same weights" either way.
        num_predict: Token cap per completion, sent as ``max_tokens``. Bounds a
            runaway draft — an uncapped hard-tier answer has no natural
            stopping point — and makes the cost of a scaled pass calculable in
            advance (Section 3.2).
        require_models: Model ids ``check-vllm`` demands from the servers.
            ``None`` means "the role models above", which is the honest
            default — those are exactly the models a full run would need.
    """

    gold_draft_base_url: str = DEFAULT_GOLD_DRAFT_BASE_URL
    gold_draft_model: str = DEFAULT_GOLD_DRAFT_MODEL
    groundedness_base_url: str = DEFAULT_GROUNDEDNESS_BASE_URL
    groundedness_model: str = DEFAULT_GROUNDEDNESS_MODEL
    sql_base_url: str = DEFAULT_GROUNDEDNESS_BASE_URL
    sql_model: str = DEFAULT_GROUNDEDNESS_MODEL
    max_parallel_calls: int = 4
    max_retries: int = 5
    backoff_base_seconds: float = 2.0
    temperature: float = 0.0
    seed: int = 7
    num_predict: int = 512
    require_models: tuple[str, ...] | None = None

    @property
    def role_models(self) -> tuple[tuple[str, str, str], ...]:
        """Returns the ``(base_url, model_name, role)`` triples a full run uses."""
        gold = (self.gold_draft_base_url, self.gold_draft_model, "gold_draft")
        ground = (
            self.groundedness_base_url,
            self.groundedness_model,
            "groundedness_check",
        )
        sql = (self.sql_base_url, self.sql_model, "sql_verification")
        roles = [gold, ground]
        if sql[:2] not in (gold[:2], ground[:2]):
            roles.append(sql)
        return tuple(roles)

    @property
    def required_model_names(self) -> tuple[str, ...]:
        """Returns the model ids ``check-vllm`` must find on their servers."""
        if self.require_models:
            return tuple(self.require_models)
        return tuple(model for _url, model, _role in self.role_models)


def get_vllm_params(args: Any, scale_config: ScaleConfig | None = None) -> VllmConfig:
    """Constructs a VllmConfig from parsed args, with scale_config as fallback.

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
        VllmConfig populated from args and, where relevant, scale_config.
    """
    if args.max_parallel_model_calls is not None:
        max_parallel = args.max_parallel_model_calls
    elif scale_config is not None:
        max_parallel = scale_config.max_parallel_model_calls
    else:
        max_parallel = VllmConfig.max_parallel_calls

    return VllmConfig(
        gold_draft_base_url=args.gold_draft_base_url,
        gold_draft_model=args.gold_draft_model,
        groundedness_base_url=args.groundedness_base_url,
        groundedness_model=args.groundedness_model,
        sql_base_url=(
            args.groundedness_base_url
            if args.sql_base_url is None
            else args.sql_base_url
        ),
        sql_model=(
            args.groundedness_model if args.sql_model is None else args.sql_model
        ),
        max_parallel_calls=max_parallel,
        max_retries=(
            VllmConfig.max_retries if args.max_retries is None else args.max_retries
        ),
        backoff_base_seconds=(
            VllmConfig.backoff_base_seconds
            if args.backoff_base_seconds is None
            else args.backoff_base_seconds
        ),
        temperature=(
            VllmConfig.temperature if args.temperature is None else args.temperature
        ),
        seed=(VllmConfig.seed if args.seed is None else args.seed),
        num_predict=(
            VllmConfig.num_predict if args.num_predict is None else args.num_predict
        ),
        require_models=tuple(args.require_models) if args.require_models else None,
    )
