"""LogRouter Evaluation Dataset — Main Entry Point.

The only executable entry point in the project, and the only function in this file
is ``main()``. Everything beside it is a library imported from here:

  src/commands/generate.py         →  generation (run_generation, generate_all_tiers, ...)
  src/commands/validate.py         →  validation (run_validation, validate_records)
  src/commands/sql_verification.py →  independent answer check by model-written SQL
  src/commands/analyzer_export.py  →  dataset out in the analyzer payload format
  src/utils/                       →  connectivity check, review lifecycle, clients

``main()`` does exactly two things, in this order. First it reads the parsed
namespace once and distributes it into every ``src.params`` dataclass, one named
local each. Then it dispatches on ``--command``, passing the dataclasses a branch
needs and nothing else.

The distribution happens up front, for all commands rather than per branch, because
that block is the complete list of what a run's configuration consists of: reading it
tells you every dataclass a parameter can reach, without following a call chain. It
is cheap — each ``get_*_params`` copies fields off a namespace and touches no corpus,
no database and no model — and the parser has already rejected an invalid combination
by this point, so nothing here can fail on a configuration a branch would not have
used anyway.

Every parameter is declared exactly once, in ``config/args.py``, and reaches a
library only through the dataclass built below. Nothing reads a value off the
namespace and re-defaults it inline, so for any parameter the one place that defines
it and its default is its owning dataclass.

Usage examples, with the container-internal paths from docker/compose.yml:

  python3 main.py --command check-vllm
  python3 main.py --command generate
  python3 main.py --command generate --full
  python3 main.py --command validate --strict
  python3 main.py --command verify-answers --sql_limit 5
  python3 main.py --command review-export
  python3 main.py --command review-apply --reviewer ada
  python3 main.py --command export-analyzer

Run ``python3 main.py --help`` to see every parameter and override.
"""

from config.args import args_parser
from src.commands.analyzer_export import run_analyzer_export
from src.commands.generate import run_generation
from src.params.corpus_params import get_corpus_params
from src.params.export_params import get_analyzer_export_params
from src.params.generation_params import get_generation_params
from src.params.results_params import build_config_snapshot
from src.params.review_params import get_review_apply_params, get_review_export_params
from src.params.scale_params import get_scale_params
from src.params.sql_verification_params import get_sql_verification_params
from src.params.validation_params import get_validation_params
from src.params.vllm_params import get_vllm_params
from src.utils.helper_vllm import check_server
from src.utils.helper_review import apply_worksheet, export_worksheet
from src.commands.validate import run_validation
from src.commands.sql_verification import run_sql_verification


def main() -> int:
    """Reads the arguments into every config dataclass, then dispatches on the command.

    Returns:
        The command's process exit code, which every branch returns rather than
        raising: the drivers in ``scripts/`` read these, and both ``validate`` and
        ``verify-answers`` distinguish "failed" from "found nothing to check" by code.

    Raises:
        ValueError: If the command is not one ``config.args`` allows. Unreachable
            through the CLI, where argparse rejects it first; reachable when
            ``main()`` is driven from a script that built its own namespace.
    """
    args = args_parser()

    corpus_config = get_corpus_params(args)
    generation_config = get_generation_params(args)
    scale_config = get_scale_params(args)
    vllm_config = get_vllm_params(args, scale_config)
    validation_config = get_validation_params(args)
    sql_verification_config = get_sql_verification_params(args)
    review_export_config = get_review_export_params(args)
    review_apply_config = get_review_apply_params(args)
    export_config = get_analyzer_export_params(args)
    config_snapshot = build_config_snapshot(args)

    if args.command == "check-vllm":
        return check_server(vllm_config)

    elif args.command == "generate":
        return run_generation(
            corpus_config, generation_config, scale_config, vllm_config
        )

    elif args.command == "validate":
        return run_validation(validation_config, config_snapshot)

    elif args.command == "verify-answers":
        return run_sql_verification(sql_verification_config, vllm_config)

    elif args.command == "review-export":
        return export_worksheet(review_export_config)

    elif args.command == "review-apply":
        return apply_worksheet(review_apply_config)

    elif args.command == "export-analyzer":
        return run_analyzer_export(export_config)

    else:
        raise ValueError(
            f"Unknown command: {args.command}. Choose check-vllm | generate | "
            f"validate | verify-answers | review-export | review-apply | export-analyzer"
        )


if __name__ == "__main__":
    raise SystemExit(main())
