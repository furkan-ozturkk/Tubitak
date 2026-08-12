"""Command-line surface for the whole project.

``args_parser()`` is the single place a run's configuration enters the system.
Everything downstream — the ``src.params`` dataclasses, the paths every command
reads and writes, the config snapshot recorded in the validation report — is
derived from the namespace this module returns, so a flag that is not defined
here cannot influence a run.

This module and the ``src.params`` dataclasses together are the project's single
configuration source. There is no YAML: a scaling parameter is a flag with a
dataclass default, which is what Section 3.2 asked for (rescaling without a code
change) without a second file that has to be edited, mounted and kept in sync.
The one non-Python config artifact in the repo is ``config/question_schema.json``,
and it exists only because ``jsonschema`` has to be handed a schema document.

Three validations run before the namespace is handed back, and all three exist to
turn a silent wrong answer into an immediate failure.

``_resolve_paths()`` derives the flags that default to another flag's value, so a
command never reads one file and writes a different one by accident.

``_validate_model_separation()`` rejects a run whose gold-draft model equals its
groundedness-check model. That pair is the Scientific Integrity Rule of Section
5.5/6: the model that writes an answer must never be the model that certifies it,
and now that both are flags, this check is the only thing standing between an
operator and a self-reviewing dataset.

``_validate_tier_knobs()`` rejects out-of-range curation knobs before any corpus
is read or any model is called, rather than letting a zero-sized evidence window
produce records that only fail at validation time.

Defaults live on the ``src.params`` dataclasses wherever a tier or a client owns
them; an ``argparse`` default of ``None`` marks a field where the dataclass
default is meant to win (see ``src.params.generation_params._resolve``).
"""

import argparse
import datetime
import os
from pathlib import Path

DEFAULT_CORPUS_DIR = Path("/data/loghub")
DATASET_PATTERN = "/output/pilot/questions_{date}.json"
FULL_DATASET_PATTERN = "/output/pilot/questions_full_{date}.json"
DEFAULT_REVIEW_DIR = Path("/output/pilot/review/groundedness")
DEFAULT_WORKSHEET = Path("/output/pilot/review/worksheet.csv")
DEFAULT_REVIEW_LOG = Path("/output/pilot/review/review_events.json")
DEFAULT_REPORT = Path("/output/pilot/validation_report.json")
DEFAULT_SCHEMA = Path(__file__).parent / "question_schema.json"
DEFAULT_MANIFEST = (
    Path(__file__).parent.parent / "src" / "corpus" / "corpus_manifest.json"
)

COMMANDS = (
    "check-vllm",
    "generate",
    "validate",
    "verify-answers",
    "review-export",
    "review-apply",
    "export-analyzer",
)


def _default_dataset() -> Path:
    """Returns today's dated path for the official dataset.

    The official output used to be one fixed ``questions.json``, so every
    write silently replaced whatever a human had already reviewed with no
    trace of what it overwrote and no way to tell, from the filename alone,
    when a given set of answers was produced. Stamping today's date into the
    default the same way ``--full`` already does closes both gaps: distinct
    days keep distinct files, and a run's date is legible from its path
    rather than only from ``gold_provenance.created_at`` inside it. Two
    generate passes on the same day still share one file — the date, not a
    full timestamp, is what was asked for.

    Returns:
        ``/output/pilot/questions_<YYYY-MM-DD>.json`` for today.
    """
    return Path(DATASET_PATTERN.format(date=datetime.date.today().isoformat()))


def _default_full_dataset() -> Path:
    """Returns today's dated scratch path for a ``--full`` generation pass.

    Every ``--full`` run used to land on the same fixed ``questions_full.json``,
    so a second experimental pass silently overwrote the first with no trace of
    what it replaced — the reason ad-hoc, hand-named backups kept accumulating
    next to it. Stamping today's date into the default path instead means
    successive full passes on different days keep their own file; two passes on
    the same day still share one (the date, not a full timestamp, is what was
    asked for), which is still a strict improvement over one name forever.

    Returns:
        ``/output/pilot/questions_full_<YYYY-MM-DD>.json`` for today.
    """
    return Path(FULL_DATASET_PATTERN.format(date=datetime.date.today().isoformat()))


def _resolve_paths(args: argparse.Namespace) -> None:
    """Fills in the flags whose correct default is another flag's resolved value.

    ``--dataset`` moves to its own dated file under ``--full``, unless the
    operator named a path explicitly. A full pass produces medium and hard
    records that leave generation ``review_status=in_review``, and the default
    target is the pilot dataset, whose model-drafted records a human may since
    have reviewed: writing one over the other would replace reviewed gold with
    fresh drafts, in place, with no copy of what it replaced. Passing
    ``--dataset`` is still honoured, so overwriting stays possible but has to be
    asked for.

    ``--questions`` is validate's input pattern list. Left unset it must be the
    dataset this invocation resolved, otherwise a validate run would certify a stale
    file while the operator believed it had checked the new one.

    ``--review_out`` is review-apply's output. Left unset it must be ``--dataset``
    itself, because applying a worksheet is an in-place edit of the dataset the
    worksheet was exported from.

    Args:
        args: Parsed namespace; the three fields are set in place when omitted.
    """
    if args.full and args.dataset == _default_dataset():
        args.dataset = _default_full_dataset()
    if args.questions is None:
        args.questions = [str(args.dataset)]
    if args.review_out is None:
        args.review_out = args.dataset


def _validate_model_separation(args: argparse.Namespace) -> None:
    """Rejects a run whose drafting and reviewing model are the same.

    Section 5.5/6 requires the model that drafts a gold answer and the model that
    runs the claim-by-claim groundedness check to be different families. Both are
    flags so an operator can move either role onto another server's catalogue,
    which means the separation can no longer be guaranteed by the source alone. It
    is enforced here, before the first prompt is sent, rather than discovered
    afterwards in a dataset whose gold answers certified themselves.

    Args:
        args: Parsed namespace carrying ``gold_draft_model`` and
            ``groundedness_model``.

    Raises:
        ValueError: If the two models are identical.
    """
    if args.gold_draft_model == args.groundedness_model:
        raise ValueError(
            f"--gold_draft_model and --groundedness_model are both "
            f"'{args.gold_draft_model}'. Section 5.5/6 requires the drafting model "
            f"and the reviewing model to be different families; a model cannot "
            f"certify the groundedness of its own answer."
        )


def _validate_tier_knobs(args: argparse.Namespace) -> None:
    """Validates the curation knobs before any corpus is read.

    Every knob checked here defaults to ``None`` so that its owning dataclass
    supplies the value; only an explicitly passed one is validated. The bounds
    catch the settings that would otherwise produce structurally invalid records —
    an empty evidence window, a question with no cited lines — which the schema
    validator would reject only after a full generation pass had already spent its
    model calls.

    Args:
        args: Parsed namespace carrying the tier knobs, the scaling knobs and
            ``test_fraction``.

    Raises:
        ValueError: If a knob is set and out of range.
    """
    positive = {
        "max_cited_lines": args.max_cited_lines,
        "window_size": args.window_size,
        "questions_per_dataset": args.questions_per_dataset,
        "min_sentences": args.min_sentences,
        "hard_pairs_per_dataset": args.hard_pairs_per_dataset,
        "hard_evidence_per_side": args.hard_evidence_per_side,
        "max_retries": args.max_retries,
        "max_parallel_model_calls": args.max_parallel_model_calls,
        "target_total_questions": args.target_total_questions,
        "sql_limit": args.sql_limit,
        "num_predict": args.num_predict,
    }
    for name, value in positive.items():
        if value is not None and value < 1:
            raise ValueError(f"--{name} must be >= 1 (got {value}).")

    if args.min_matches is not None and args.min_matches < 0:
        raise ValueError(f"--min_matches must be >= 0 (got {args.min_matches}).")

    if args.test_fraction is not None and not 0.0 <= args.test_fraction < 1.0:
        raise ValueError(
            f"--test_fraction must satisfy 0.0 <= f < 1.0 (got {args.test_fraction}); "
            f"1.0 would put every question in the test split and leave no dev set."
        )


def args_parser(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for generation, validation and review.

    Args:
        argv: Argument list to parse; ``None`` reads ``sys.argv``.

    Returns:
        Parsed argument namespace with all path, tier, scaling and model fields.
    """
    parser = argparse.ArgumentParser(
        description="LogRouter evaluation dataset — generation / validation / review",
    )

    parser.add_argument(
        "--command",
        type=str,
        default="generate",
        choices=list(COMMANDS),
        help="Operation to run: check-vllm (connectivity + required models, Section 5.5/6), "
        "generate (write the question dataset, Section 3.1/7), validate (schema + cross-record "
        "+ evidence checks, Sections 2/6), verify-answers (independent check: a model writes "
        "the SQL for each question and its result is compared to the gold answer), "
        "review-export (export in_review records to a CSV worksheet), review-apply (apply a "
        "filled-in worksheet back onto the dataset), export-analyzer (write the dataset in the "
        "LLM Log Analyzer evaluation payload format)",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=_default_dataset(),
        help="The question dataset file: written by generate, read by review-export, read and "
        f"rewritten by review-apply, and the default input of validate. Defaults to today's "
        f"dated path ({DATASET_PATTERN.format(date='YYYY-MM-DD')}) so distinct days never "
        "overwrite each other; pass this explicitly to operate on a different day's file. "
        "Under --full the default moves instead to a dated scratch file "
        f"({FULL_DATASET_PATTERN.format(date='YYYY-MM-DD')}) so an experimental pass never "
        "overwrites the same day's official output unless this flag says so explicitly",
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory holding the fetched LogHub *_2k.log files (mounted read-only from the "
        "loghub volume)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="question_schema.json path (Section 6 single source of truth)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="corpus_manifest.json path. validate uses it to require every pinned dataset "
        "to be represented; export-analyzer uses it to name each dataset's corpus file in the "
        "exported payload",
    )
    parser.add_argument(
        "--review_dir",
        type=Path,
        default=DEFAULT_REVIEW_DIR,
        help="Directory the hard tier writes its per-question groundedness reports to, and "
        "review-export summarises from",
    )
    parser.add_argument(
        "--worksheet",
        type=Path,
        default=DEFAULT_WORKSHEET,
        help="CSV worksheet review-export writes and review-apply reads",
    )
    parser.add_argument(
        "--review_log",
        type=Path,
        default=DEFAULT_REVIEW_LOG,
        help="[review-apply] Append-only JSON log of review decisions: who decided what, when, "
        "and against which draft digest",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="JSON validation report written by the validate command",
    )
    parser.add_argument(
        "--sql_report",
        type=Path,
        default=None,
        help="[verify-answers] JSON report path (dataclass default "
        "/output/pilot/sql_verification_report.json)",
    )
    parser.add_argument(
        "--sql_limit",
        type=int,
        default=None,
        help="[verify-answers] Check at most this many records; each costs one model call. "
        "Omit to check every eligible record",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=None,
        help="[validate] JSON/JSONL file(s) or glob pattern(s) to validate. Omit to validate "
        "--dataset",
    )
    parser.add_argument(
        "--review_out",
        type=Path,
        default=None,
        help="[review-apply] Where the updated dataset is written. Omit to overwrite --dataset "
        "in place",
    )
    parser.add_argument(
        "--export_out",
        type=Path,
        default=None,
        help="[export-analyzer] Where the analyzer-format payload is written (dataclass "
        "default /output/pilot/questions_analyzer.json)",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="[generate] Same full three-tier pass, but written to its own dated scratch file "
        f"({FULL_DATASET_PATTERN.format(date='YYYY-MM-DD')} unless --dataset overrides it) so "
        "an experimental run never overwrites a pilot dataset whose model-drafted records a "
        "human has since reviewed, and successive full passes on different days do not "
        "overwrite each other either",
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=None,
        help="[generate/easy] Prune count literals with fewer than this many real matches "
        "(dataclass default 3)",
    )
    parser.add_argument(
        "--max_cited_lines",
        type=int,
        default=None,
        help="[generate/easy] Cap on evidence.refs per count/presence question (dataclass "
        "default 5)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="[generate/medium] Evidence lines per medium question; Section 7.2 caps it at 8 "
        "(dataclass default 8)",
    )
    parser.add_argument(
        "--questions_per_dataset",
        type=int,
        default=None,
        help="[generate/medium] Anchor occurrences drafted per LogHub dataset (dataclass "
        "default 2)",
    )
    parser.add_argument(
        "--min_sentences",
        type=int,
        default=None,
        help="[generate/hard] Minimum sentences demanded of a hard gold answer; Section 7.3 "
        "expects >=4 (dataclass default 4)",
    )
    parser.add_argument(
        "--hard_pairs_per_dataset",
        type=int,
        default=None,
        help="[generate/hard] Non-overlapping group-sets drafted per hard_groups spec "
        "(dataclass default 1)",
    )
    parser.add_argument(
        "--hard_evidence_per_side",
        type=int,
        default=None,
        help="[generate/hard] A group's first and last this-many lines are cited (all of "
        "it when the group has <= 2x this many) (dataclass default 15)",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=None,
        help="Fraction of evidence groups hashed into the test split; the rest go to dev "
        "(dataclass default 0.20)",
    )
    parser.add_argument(
        "--reviewer",
        type=str,
        default=None,
        help="Who is responsible for this run's records. On generate it is written into every "
        "record's reviewers[] (dataclass default faz1_pilot_script, the generating script); on "
        "review-apply it must name the human making the accept/edit/reject decisions and is "
        "recorded in the review event log",
    )
    parser.add_argument(
        "--created_at",
        type=str,
        default=None,
        help="Fixed gold_provenance.created_at timestamp; held constant so repeated runs are "
        "byte-identical (Section 6 determinism, dataclass default 2026-08-01T00:00:00Z)",
    )

    parser.add_argument(
        "--target_total_questions",
        type=int,
        default=None,
        help="Question target for the pass, reported beside the realised counts and never "
        "enforced; raise it to scale a run without a code change (Section 3.2, dataclass "
        "default 100)",
    )

    parser.add_argument(
        "--gold_draft_base_url",
        type=str,
        default=os.environ.get("GOLD_DRAFT_BASE_URL", "http://furkan.ozturk_vllm:8003"),
        help="Local vLLM server that drafts medium/hard gold answers (Section 5.5)",
    )
    parser.add_argument(
        "--gold_draft_model",
        type=str,
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        help="Model id --gold_draft_base_url was started with. Must differ from "
        "--groundedness_model (Section 5.5/6)",
    )
    parser.add_argument(
        "--groundedness_base_url",
        type=str,
        default=os.environ.get(
            "GROUNDEDNESS_BASE_URL", "http://furkan.ozturk_vllm:8001"
        ),
        help="Local vLLM server that runs the claim-by-claim groundedness check "
        "(Section 5.5)",
    )
    parser.add_argument(
        "--groundedness_model",
        type=str,
        default="Qwen/Qwen3-32B-AWQ",
        help="Model id --groundedness_base_url was started with. Must differ from "
        "--gold_draft_model (Section 5.5/6)",
    )
    parser.add_argument(
        "--sql_base_url",
        type=str,
        default=None,
        help="[verify-answers] Server that writes the verification SQL. Omit to reuse "
        "--groundedness_base_url",
    )
    parser.add_argument(
        "--sql_model",
        type=str,
        default=None,
        help="[verify-answers] Model id --sql_base_url was started with. Omit to reuse "
        "--groundedness_model, which is already a different family from the drafting model, "
        "so the query checking an answer never comes from the model that wrote it",
    )
    parser.add_argument(
        "--require_models",
        nargs="*",
        default=None,
        help="[check-vllm] Model ids that must be served. Omit to require the "
        "role models above",
    )
    parser.add_argument(
        "--max_parallel_model_calls",
        type=int,
        default=None,
        help="Concurrent vLLM calls (dataclass default 4)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="Attempts per vLLM call before giving up (dataclass default 5)",
    )
    parser.add_argument(
        "--backoff_base_seconds",
        type=float,
        default=None,
        help="Base of the exponential retry backoff, in seconds (dataclass default 2.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for every model call; 0.0 keeps drafts reproducible "
        "(dataclass default 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed sent with every model call; pinned so greedy decoding stays "
        "deterministic across server restarts (dataclass default 7)",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=None,
        help="Token cap per model completion, sent as max_tokens, bounding runaway "
        "drafts (dataclass default 512)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="[validate] Turn warnings that Section 2/7.4 states as rules — currently the >=3 "
        "phrasing-families-per-intent rule — into errors",
    )

    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=None,
        help="Optional free-form tag identifying this run; recorded in the validation report's "
        "config snapshot",
    )

    args = parser.parse_args(argv)
    _resolve_paths(args)
    _validate_model_separation(args)
    _validate_tier_knobs(args)
    return args
