#!/usr/bin/env python3
"""
main.py

Single entry point for the datasetgen container. Everything that used to be
a separate script (check_ollama.py, build_pilot.py, schema/validate_schema.py,
review_worksheet.py) is now one subcommand each.

Every parameter this program accepts is declared exactly once, in cli.py,
as a frozen dataclass per subcommand (CheckOllamaArgs, GenerateArgs, ...);
parse_args() converts the raw argparse.Namespace into that dataclass before
any command function below ever sees it. scale_config.yaml is read the same
way, into scale_config.ScaleConfig's RunConfig/DifficultyMix/ConcurrencyConfig.
So for any parameter, its owning dataclass in cli.py or scale_config.py is
the one place that defines it and its default -- nothing here re-declares
or re-defaults a parameter inline.

Usage (paths below are the container-internal paths from docker-compose.yml):

  docker compose exec datasetgen python3 main.py check-ollama
  docker compose exec datasetgen python3 main.py generate
  docker compose exec datasetgen python3 main.py validate
  docker compose exec datasetgen python3 main.py review-export
  docker compose exec datasetgen python3 main.py review-apply

Run with --help on a subcommand to see its parameters/overrides.
"""
import json
import sys
import time
import urllib.request

import human_review
import question_generators
import schema_validator
from cli import CheckOllamaArgs, GenerateArgs, ReviewApplyArgs, ReviewExportArgs, ValidateArgs, parse_args
from ollama_client import OllamaClient
from scale_config import ScaleConfig

REQUIRED_OLLAMA_MODELS = [
    {"name": "nemotron-3-nano:30b", "role": "gold_draft"},
    {"name": "gpt-oss:20b", "role": "groundedness_check"},
]


# --------------------------------------------------------------------------
# check-ollama (Section 5.5/6): connectivity + required-model check
# --------------------------------------------------------------------------
def _get_with_retry(url: str, max_retries: int = 5, backoff_base_seconds: float = 2.0) -> dict:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "logrouter-datasetgen/main"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                wait = backoff_base_seconds * (2 ** (attempt - 1))
                print(f"  [retry {attempt}/{max_retries}] {url} -> {e} ; waiting {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"Connection failed ({max_retries} attempts): {url} :: {last_err}")


def cmd_check_ollama(args: CheckOllamaArgs) -> int:
    tags_url = args.base_url.rstrip("/") + "/api/tags"
    print(f"Connecting to Ollama server: {tags_url}")

    try:
        payload = _get_with_retry(tags_url)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Note: this infrastructure is outside this project's control (Section 5.5); if the "
              "problem persists, check the server address/port or network reachability.", file=sys.stderr)
        return 2

    available = {m.get("name") for m in payload.get("models", [])}
    print(f"Models on server ({len(available)}): {sorted(available)}")

    required_names = args.require_models or [m["name"] for m in REQUIRED_OLLAMA_MODELS]
    missing = [m for m in required_names if m not in available]

    # Make sure the same model family isn't used for both drafting and review (Section 5.5/6).
    if len(set(required_names)) != len(required_names):
        print("ERROR: required model list contains a duplicate; the drafting and reviewing model "
              "must never be the same.", file=sys.stderr)
        return 3

    if missing:
        print(f"ERROR: required model(s) missing from server: {missing}", file=sys.stderr)
        return 1

    print("OK: connection established and all required models are present on the server.")
    for m in REQUIRED_OLLAMA_MODELS:
        print(f"  - {m['name']:<24} role={m['role']}")
    return 0


# --------------------------------------------------------------------------
# generate (Section 3.1): default = official 20-question stage-1 set (easy
# tier only, no model); --full = all three tiers, merged into one dataset
# --------------------------------------------------------------------------
def cmd_generate(args: GenerateArgs) -> int:
    if not args.full:
        print("=== Official 20-question set (easy tier only, no model) ===")
        records = question_generators.generate_official_20(args.corpus_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== SUMMARY ===\ntotal={len(records)} (official stage-1 set)")
        print(f"Wrote dataset to {args.out}")
        return 0

    config = ScaleConfig.load(args.config)

    print("=== Easy (deterministic) ===")
    easy_records = question_generators.generate_easy(args.corpus_dir)

    client = OllamaClient(max_parallel_calls=config.concurrency.max_parallel_model_calls)

    print("\n=== Medium (semantic) ===")
    medium_records = question_generators.generate_medium(args.corpus_dir, client)

    print("\n=== Hard ===")
    hard_records = question_generators.generate_hard(args.corpus_dir, client, args.review_dir)

    all_records = easy_records + medium_records + hard_records
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(all_records)
    print("\n=== SUMMARY ===")
    print(f"easy={len(easy_records)} medium={len(medium_records)} hard={len(hard_records)} "
          f"total={total} (target ~{config.run.target_total_questions}, "
          f"configured mix={config.difficulty_mix})")
    print(f"Wrote dataset to {args.out}")
    return 0


# --------------------------------------------------------------------------
# validate (Sections 2/6)
# --------------------------------------------------------------------------
def cmd_validate(args: ValidateArgs) -> int:
    return schema_validator.run(
        questions_patterns=args.questions,
        schema_path=args.schema,
        corpus_dir=args.corpus_dir,
        manifest=args.manifest,
        strict=args.strict,
        report_path=args.report,
    )


# --------------------------------------------------------------------------
# review-export / review-apply (Section 7.3 step 5 / Section 6)
# --------------------------------------------------------------------------
def cmd_review_export(args: ReviewExportArgs) -> int:
    return human_review.export(args.questions, args.worksheet, args.review_dir)


def cmd_review_apply(args: ReviewApplyArgs) -> int:
    return human_review.apply(args.questions, args.worksheet, args.out)


_DISPATCH = {
    "check-ollama": cmd_check_ollama,
    "generate": cmd_generate,
    "validate": cmd_validate,
    "review-export": cmd_review_export,
    "review-apply": cmd_review_apply,
}


def main() -> int:
    command, command_args = parse_args()
    return _DISPATCH[command](command_args)


if __name__ == "__main__":
    raise SystemExit(main())
