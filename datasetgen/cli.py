#!/usr/bin/env python3
"""
cli.py

Single home for every CLI parameter main.py accepts. Argparse only fills in
raw strings/Paths; each subcommand's parameters are then handed to their own
dataclass (declared here) before any command function sees them, so a
parameter always has one obvious owner instead of being read ad hoc off a
generic argparse.Namespace deep inside a command function.

parse_args() is the only function main.py calls: it returns the subcommand
name plus the one dataclass instance that command's parameters live in.
"""
import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG = Path("/app/scale_config.yaml")
DEFAULT_CORPUS_DIR = Path("/data/loghub")
DEFAULT_OUT = Path("/output/pilot/questions.json")
DEFAULT_REVIEW_DIR = Path("/output/pilot/review/groundedness")
DEFAULT_WORKSHEET = Path("/output/pilot/review/worksheet.csv")
DEFAULT_REPORT = Path("/output/pilot/validation_report.json")
DEFAULT_SCHEMA = Path(__file__).parent / "question_schema.json"


@dataclass(frozen=True)
class CheckOllamaArgs:
    """`main.py check-ollama` (Section 5.5/6): connectivity + required-model check."""
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://10.15.33.66:11435")
    require_models: Optional[list] = None


@dataclass(frozen=True)
class GenerateArgs:
    """`main.py generate` (Section 3.1): by default writes the official
    20-question stage-1 set (easy tier only, no model); --full runs all
    three tiers (easy+medium+hard) instead."""
    config: Path = DEFAULT_CONFIG
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    out: Path = DEFAULT_OUT
    review_dir: Path = DEFAULT_REVIEW_DIR
    full: bool = False


@dataclass(frozen=True)
class ValidateArgs:
    """`main.py validate` (Sections 2/6): schema + cross-record + evidence/groundedness checks."""
    questions: list = field(default_factory=lambda: [str(DEFAULT_OUT)])
    schema: Path = DEFAULT_SCHEMA
    corpus_dir: Path = DEFAULT_CORPUS_DIR
    manifest: Optional[Path] = None
    strict: bool = False
    report: Path = DEFAULT_REPORT


@dataclass(frozen=True)
class ReviewExportArgs:
    """`main.py review-export`: export in_review records to a CSV worksheet."""
    questions: Path = DEFAULT_OUT
    worksheet: Path = DEFAULT_WORKSHEET
    review_dir: Path = DEFAULT_REVIEW_DIR


@dataclass(frozen=True)
class ReviewApplyArgs:
    """`main.py review-apply`: apply a filled-in worksheet back onto the questions file."""
    questions: Path = DEFAULT_OUT
    worksheet: Path = DEFAULT_WORKSHEET
    out: Optional[Path] = None  # None means "overwrite questions"


# Maps each subcommand name to (the dataclass that owns its parameters,
# its --help text, the function that adds its argparse flags).
_COMMANDS = {}


def _register(name, help_text, dataclass_type, add_arguments):
    _COMMANDS[name] = (dataclass_type, help_text, add_arguments)


def _add_check_ollama_arguments(p):
    p.add_argument("--base-url", default=CheckOllamaArgs.base_url)
    p.add_argument("--require-models", nargs="*", default=None,
                    help="model names that must be present on the server")


def _add_generate_arguments(p):
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    p.add_argument("--full", action="store_true",
                    help="run all three tiers (easy+medium+hard, needs Ollama) instead of "
                         "the default 20-question easy-only stage-1 set")


def _add_validate_arguments(p):
    p.add_argument("--questions", nargs="+", default=[str(DEFAULT_OUT)],
                    help="JSON/JSONL file(s) or glob pattern(s)")
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--strict", action="store_true",
                    help="turn some warnings, such as phrasing diversity, into errors")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)


def _add_review_export_arguments(p):
    p.add_argument("--questions", type=Path, default=DEFAULT_OUT)
    p.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    p.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)


def _add_review_apply_arguments(p):
    p.add_argument("--questions", type=Path, default=DEFAULT_OUT)
    p.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    p.add_argument("--out", type=Path, default=None, help="defaults to overwriting --questions")


_register("check-ollama", "verify Ollama connectivity and required models (Section 5.5/6)",
           CheckOllamaArgs, _add_check_ollama_arguments)
_register("generate", "run all three tiers and write the merged dataset (Section 3.1)",
           GenerateArgs, _add_generate_arguments)
_register("validate", "schema + cross-record + evidence/groundedness validation (Sections 2/6)",
           ValidateArgs, _add_validate_arguments)
_register("review-export", "export in_review records to a CSV worksheet",
           ReviewExportArgs, _add_review_export_arguments)
_register("review-apply", "apply a filled-in worksheet back onto the questions file",
           ReviewApplyArgs, _add_review_apply_arguments)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Single entry point for the datasetgen container "
                     "(see each subcommand's --help for its parameters).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (_dataclass_type, help_text, add_arguments) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        add_arguments(p)
        p.set_defaults(command=name)
    return ap


def parse_args(argv=None):
    """Parse argv and return (command_name, <the dataclass instance owning that command's args>)."""
    args = build_parser().parse_args(argv)
    dataclass_type, _help_text, _add_arguments = _COMMANDS[args.command]
    field_names = set(dataclass_type.__dataclass_fields__)
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return args.command, dataclass_type(**kwargs)
