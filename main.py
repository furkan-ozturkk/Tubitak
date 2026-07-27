"""LogRouter Evaluation Dataset — Main Entry Point.

The only executable entry point in the project, and the only function in this file
is ``main()``. ``generate.py`` and ``validate.py`` are libraries imported from
here, not scripts of their own:

  generate.py  →  generation (run_generation, generate_easy, generate_full, ...)
  validate.py  →  validation  (run_validation, validate_records)
  main.py      →  entry point that wires everything together

Each command's work lives in the library that owns it, so this file holds a
dispatch and nothing else: connectivity checking is Ollama's concern
(``src.utils.helper_ollama.check_server``), the review lifecycle is
``src.utils.helper_review``'s. Anything that grew a helper function here would be
logic that had escaped the module responsible for it.

Every parameter a run accepts is declared exactly once, in ``config/args.py``, and
reaches a library only through a ``src.params`` dataclass built by a
``get_*_params(args)`` call below. Nothing reads a value off the namespace and
re-defaults it inline, so for any parameter the one place that defines it and its
default is its owning dataclass.

Usage examples, with the container-internal paths from docker/compose.yml:

  python3 main.py --command check-ollama
  python3 main.py --command generate
  python3 main.py --command generate --full
  python3 main.py --command validate --strict
  python3 main.py --command review-export
  python3 main.py --command review-apply --reviewer ada

Run ``python3 main.py --help`` to see every parameter and override.
"""

from config.args import args_parser
from generate import run_generation
from src.params.corpus_params import get_corpus_params
from src.params.generation_params import get_generation_params
from src.params.ollama_params import get_ollama_params
from src.params.results_params import build_config_snapshot
from src.params.review_params import get_review_apply_params, get_review_export_params
from src.params.scale_params import get_scale_params
from src.params.validation_params import get_validation_params
from src.utils.helper_ollama import check_server
from src.utils.helper_review import apply_worksheet, export_worksheet
from validate import run_validation


def main() -> int:
    """Parses arguments and dispatches to the library that owns the command.

    Returns:
        The command's process exit code, which every branch returns rather than
        raising: the pipeline in ``scripts/`` reads these, and ``validate``
        distinguishes "failed" from "found nothing to check" by code.

    Raises:
        ValueError: If the command is not one ``config.args`` allows. Unreachable
            through the CLI, where argparse rejects it first; reachable when
            ``main()`` is driven from a script that built its own namespace.
    """
    args = args_parser()

    if args.command == "check-ollama":
        return check_server(get_ollama_params(args))

    elif args.command == "generate":
        return run_generation(
            get_corpus_params(args),
            get_generation_params(args),
            get_scale_params(args),
            get_ollama_params(args),
        )

    elif args.command == "validate":
        return run_validation(get_validation_params(args), build_config_snapshot(args))

    elif args.command == "review-export":
        return export_worksheet(get_review_export_params(args))

    elif args.command == "review-apply":
        return apply_worksheet(get_review_apply_params(args))

    else:
        raise ValueError(
            f"Unknown command: {args.command}. Choose check-ollama | generate | "
            f"validate | review-export | review-apply"
        )


if __name__ == "__main__":
    raise SystemExit(main())
