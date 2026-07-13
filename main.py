"""Single entry point for the LogRouter benchmark extension project.

Subcommands are ``ingest``, ``index``, ``route``, ``evaluate``, and
``ablate``, mirroring the two stages of the LogRouter system: the offline
indexing pipeline and the query-time routing layer. Only ``ingest`` is
currently implemented.

Reference:
    LogRouter paper, Figure 1 (indexing pipeline) and Figure 2 (query-time
    routing).
"""

import argparse
import sys

from config import Config
from storage import loghub_loader, structured_store


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser with all
        subcommands registered.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LogRouter benchmark extension entry point.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load Loghub-2.0 data into Postgres.")
    ingest_parser.add_argument(
        "--dataset",
        default="HDFS",
        help="Comma-separated Loghub-2.0 system name(s), for example HDFS or HDFS,BGL.",
    )

    subparsers.add_parser("index", help="Not implemented yet.")
    subparsers.add_parser("route", help="Not implemented yet.")
    subparsers.add_parser("evaluate", help="Not implemented yet.")
    subparsers.add_parser("ablate", help="Not implemented yet.")

    return parser


def run_ingest(config: Config, dataset_arg: str) -> int:
    """Loads one or more Loghub-2.0 systems into the ``logs`` table.

    Args:
        config: Application configuration.
        dataset_arg: Comma-separated Loghub-2.0 system name(s).

    Returns:
        int: Process exit code, ``0`` on success.
    """
    datasets = [name.strip() for name in dataset_arg.split(",") if name.strip()]
    unknown = [name for name in datasets if name not in loghub_loader.KNOWN_DATASETS]
    if unknown:
        print(f"Warning: unknown dataset name(s): {unknown}. Known: {loghub_loader.KNOWN_DATASETS}")

    conn = structured_store.connect(config)
    try:
        structured_store.ensure_schema(conn)
        total = 0
        for dataset in datasets:
            print(f"Fetching {dataset} from Loghub-2.0...")
            rows = loghub_loader.load_dataset(dataset)
            inserted = structured_store.insert_rows(conn, rows)
            total += inserted
            print(f"{dataset}: {inserted} rows processed.")
        print(f"Done. {total} rows processed in total.")
    finally:
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parses arguments and dispatches to the selected subcommand.

    Args:
        argv: Command-line arguments, or ``None`` to use ``sys.argv``.

    Returns:
        int: Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config.from_env()

    if args.command == "ingest":
        return run_ingest(config, args.dataset)

    raise NotImplementedError(f"Command '{args.command}' is not implemented yet.")


if __name__ == "__main__":
    sys.exit(main())
