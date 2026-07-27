"""Configuration parameters for the verify-answers command.

Independent verification of the deterministic answers, by a route that shares no
code with the one that produced them: a model is shown the question and the table
shape, writes the SQL itself, and its result is compared to the gold answer.

This replaces a hand-written ``verify_answers.sql``. That file listed the twenty
expected values next to the queries that produced them, so reading it told you the
generator and the file agreed — which they would, having been written from each
other. A query derived from the question alone can disagree, and a disagreement is
the only outcome that carries information.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SQL_REPORT = Path("/output/pilot/sql_verification_report.json")


@dataclass(frozen=True)
class SqlVerificationConfig:
    """What to verify and where to record the result.

    ``routing_paths`` limits the check to the records whose answers a query can
    settle. A semantic explanation has no SQL that reproduces it, so asking a model
    to write one would produce a failure about the method rather than about the
    answer.

    Attributes:
        dataset: Question dataset to verify.
        report: JSON report path.
        routing_paths: Routing paths eligible for verification.
        limit: Maximum records to check, or ``None`` for all. Each record costs one
            model call, so a smoke check over five is worth having.
    """

    dataset: Path
    report: Path = DEFAULT_SQL_REPORT
    routing_paths: tuple[str, ...] = ("sql", "keyword")
    limit: int | None = None


def get_sql_verification_params(args: Any) -> SqlVerificationConfig:
    """Constructs a SqlVerificationConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        SqlVerificationConfig populated from args, with the dataclass default
        winning wherever the flag was omitted.
    """
    return SqlVerificationConfig(
        dataset=args.dataset,
        report=(
            SqlVerificationConfig.report if args.sql_report is None else args.sql_report
        ),
        limit=args.sql_limit,
    )
