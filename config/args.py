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
import os
from pathlib import Path

DEFAULT_CORPUS_DIR = Path("/data/loghub")
DEFAULT_DATASET = Path("/output/pilot/questions.json")
DEFAULT_FULL_DATASET = Path("/output/pilot/questions_full.json")
DEFAULT_REVIEW_DIR = Path("/output/pilot/review/groundedness")
DEFAULT_WORKSHEET = Path("/output/pilot/review/worksheet.csv")
DEFAULT_REVIEW_LOG = Path("/output/pilot/review/review_events.json")
DEFAULT_REPORT = Path("/output/pilot/validation_report.json")
DEFAULT_SCHEMA = Path(__file__).parent / "question_schema.json"

COMMANDS = (
    "check-ollama",
    "generate",
    "validate",
    "verify-answers",
    "review-export",
    "review-apply",
)


def _resolve_paths(args: argparse.Namespace) -> None:
    """Fills in the flags whose correct default is another flag's resolved value.

    ``--dataset`` moves to its own file under ``--full``, unless the operator named
    a path explicitly. A full pass produces medium and hard records that leave
    generation ``review_status=in_review``, and the default target is the official
    stage-1 output whose twenty records are all ``verified``: writing one over the
    other would replace a verified dataset with drafts, in place, with no copy of
    what it replaced. Passing ``--dataset`` is still honoured, so overwriting stays
    possible but has to be asked for.

    ``--questions`` is validate's input pattern list. Left unset it must be the
    dataset this invocation resolved, otherwise a validate run would certify a stale
    file while the operator believed it had checked the new one.

    ``--review_out`` is review-apply's output. Left unset it must be ``--dataset``
    itself, because applying a worksheet is an in-place edit of the dataset the
    worksheet was exported from.

    Args:
        args: Parsed namespace; the three fields are set in place when omitted.
    """
    if args.full and args.dataset == DEFAULT_DATASET:
        args.dataset = DEFAULT_FULL_DATASET
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
        "max_retries": args.max_retries,
        "max_parallel_model_calls": args.max_parallel_model_calls,
        "target_total_questions": args.target_total_questions,
        "sql_limit": args.sql_limit,
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
        help="Operation to run: check-ollama (connectivity + required models, Section 5.5/6), "
        "generate (write the question dataset, Section 3.1/7), validate (schema + cross-record "
        "+ evidence checks, Sections 2/6), verify-answers (independent check: a model writes "
        "the SQL for each question and its result is compared to the gold answer), "
        "review-export (export in_review records to a CSV worksheet), review-apply (apply a "
        "filled-in worksheet back onto the dataset)",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="The question dataset file: written by generate, read by review-export, read and "
        "rewritten by review-apply, and the default input of validate. Under --full the default "
        f"moves to {DEFAULT_FULL_DATASET} so a three-tier draft pass never overwrites the "
        "official stage-1 output unless this flag says so explicitly",
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
        default=None,
        help="corpus_manifest.json path, recorded alongside the corpus index for provenance",
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
        "--full",
        action="store_true",
        help="[generate] Run all three tiers (easy+medium+hard, needs Ollama) instead of the "
        "default 20-question easy-only stage-1 set",
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
        "--base_url",
        type=str,
        default=os.environ.get("OLLAMA_BASE_URL", "http://10.15.33.66:11435"),
        help="Remote Ollama server (Section 5.5); its operation is outside this project's control",
    )
    parser.add_argument(
        "--gold_draft_model",
        type=str,
        default="nemotron-3-nano:30b",
        help="Model that drafts medium/hard gold answers. Must differ from --groundedness_model "
        "(Section 5.5/6)",
    )
    parser.add_argument(
        "--groundedness_model",
        type=str,
        default="gpt-oss:20b",
        help="Model that runs the claim-by-claim groundedness check. Must differ from "
        "--gold_draft_model (Section 5.5/6)",
    )
    parser.add_argument(
        "--sql_model",
        type=str,
        default=None,
        help="[verify-answers] Model that writes the verification SQL. Omit to reuse "
        "--groundedness_model, which is already a different family from the drafting model, "
        "so the query checking an answer never comes from the model that wrote it",
    )
    parser.add_argument(
        "--require_models",
        nargs="*",
        default=None,
        help="[check-ollama] Model names that must be present on the server. Omit to require the "
        "two role models above",
    )
    parser.add_argument(
        "--max_parallel_model_calls",
        type=int,
        default=None,
        help="Concurrent Ollama calls (dataclass default 4)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=None,
        help="Attempts per Ollama call before giving up (dataclass default 5)",
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
