"""Independent answer verification by model-written SQL.

A library, not a script: ``main.py`` is the only executable entry point, and
``run_sql_verification`` is what its ``verify-answers`` command calls.

The point of this route is that it shares no code with the one that produced the
answers. ``src/generators/easy_tier.py`` builds a query from a curated literal and
records the result; here a model is shown the question text and the table shape, and
writes its own query. Nothing about the gold answer, the literal, or the cited lines
reaches the prompt, so the model has to derive the answer rather than reproduce it.
Agreement then means two independent derivations landed on the same number.

The previous form of this check was a hand-written ``scripts/verify_answers.sql``
listing the twenty expected values beside the queries that produced them. Reading it
established that the generator and the file agreed, which they did by construction,
having been written from one another. It could not disagree, and a check that cannot
disagree carries no information.

Model-written SQL is untrusted input and is executed accordingly:
``helper_postgres.assert_readonly_select`` rejects anything that is not a lone
SELECT, ``run_readonly_query`` runs it in a READ ONLY transaction that Postgres
itself enforces, and the connection carries a statement timeout so an expensive
query cannot pin a backend.

A disagreement is reported, never acted on. This command does not edit the dataset:
a model's query being wrong is at least as likely as the gold being wrong, and which
one it is has to be read by a person.
"""

from typing import Any

from src.params.ollama_params import OllamaConfig
from src.params.sql_verification_params import SqlVerificationConfig
from src.utils import helper_postgres
from src.utils.helper_ollama import OllamaClient
from src.utils.helper_records import dataset_key_from_evidence, load_questions
from src.utils.helper_run import write_json


def _scalar(rows: list[tuple]) -> Any:
    """Extracts the single value a verification query should return.

    Args:
        rows: Rows returned by the query.

    Returns:
        The first column of the first row, or ``None`` when nothing was returned.
    """
    if not rows or not rows[0]:
        return None
    return rows[0][0]


def compare(answer_type: str, expected_answer: str, value: Any) -> tuple[bool, str]:
    """Compares a query result against the gold answer.

    Each answer type is compared on its own terms rather than by string equality: a
    count query returns an integer, a presence query may return a boolean or a
    count, and a lookup returns text. Normalising all three to strings would report
    ``True`` against ``"Yes"`` as a disagreement.

    Args:
        answer_type: The record's ``answer_type``.
        expected_answer: The gold answer.
        value: The scalar the query returned.

    Returns:
        Tuple ``(agrees, detail)`` where detail describes what was compared.
    """
    if value is None:
        return False, "the query returned no rows"

    if answer_type == "count":
        try:
            return (
                str(int(value)) == expected_answer,
                f"query={int(value)} gold={expected_answer}",
            )
        except (TypeError, ValueError):
            return False, f"query returned {value!r}, which is not a count"

    if answer_type == "presence":
        if isinstance(value, bool):
            present = value
        else:
            try:
                present = int(value) > 0
            except (TypeError, ValueError):
                return (
                    False,
                    f"query returned {value!r}, which is not a boolean or count",
                )
        derived = "Yes" if present else "No"
        agrees = expected_answer.startswith(derived)
        return agrees, f"query={derived} gold={expected_answer}"

    if answer_type == "line_lookup":
        text = value if isinstance(value, str) else str(value)
        return text.strip() == expected_answer.strip(), (
            "query text matches gold"
            if text.strip() == expected_answer.strip()
            else "query returned a different line"
        )

    return False, f"answer_type={answer_type} cannot be settled by a query"


def verify_record(record: dict[str, Any], client: OllamaClient) -> dict[str, Any]:
    """Verifies one record by asking the model for SQL and running it.

    Every failure mode is recorded rather than raised, because one unusable query
    must not end the pass: the report is more useful listing which records agreed
    than stopping at the first model that emitted prose.

    Args:
        record: The question record.
        client: Ollama client; the query is written by ``sql_model``.

    Returns:
        One report entry.
    """
    dataset_key = dataset_key_from_evidence(record)
    entry: dict[str, Any] = {
        "id": record.get("id"),
        "answer_type": record.get("answer_type"),
        "dataset": dataset_key,
        "expected_answer": record.get("expected_answer"),
        "agrees": False,
        "sql": None,
        "result": None,
        "detail": None,
        "model": None,
    }

    if not dataset_key:
        entry["detail"] = "the record cites no resolvable dataset"
        return entry

    try:
        written = client.write_sql(record["question"], dataset_key)
    except RuntimeError as error:
        entry["detail"] = f"the model call failed: {error}"
        return entry

    entry["sql"] = written["sql"]
    entry["model"] = written["model"]

    try:
        rows = helper_postgres.run_readonly_query(written["sql"])
    except ValueError as error:
        entry["detail"] = f"the query was refused: {error}"
        return entry
    except Exception as error:
        entry["detail"] = f"the query failed: {error}"
        return entry

    value = _scalar(rows)
    entry["result"] = (
        value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
    )
    agrees, detail = compare(
        record.get("answer_type", ""), record.get("expected_answer", ""), value
    )
    entry["agrees"] = agrees
    entry["detail"] = detail
    return entry


def run_sql_verification(
    config: SqlVerificationConfig, ollama_config: OllamaConfig
) -> int:
    """Verifies a dataset's deterministic answers and writes the report.

    Args:
        config: What to verify and where to record it.
        ollama_config: Server address and model roles.

    Returns:
        ``0`` every eligible record agreed, ``1`` at least one disagreed or could not
        be checked, ``2`` there was nothing eligible to check.
    """
    print("Command      : verify-answers")
    print(f"Dataset      : {config.dataset}")
    print(f"SQL model    : {ollama_config.sql_model}")

    try:
        records = load_questions([str(config.dataset)])
        eligible = [
            record
            for record in records
            if record.get("routing_path") in config.routing_paths
        ]
        if config.limit is not None:
            eligible = eligible[: config.limit]

        if not eligible:
            print(
                f"Nothing to verify: no records with routing_path in "
                f"{list(config.routing_paths)}."
            )
            return 2

        client = OllamaClient(ollama_config)
        entries = []
        for record in eligible:
            entry = verify_record(record, client)
            entries.append(entry)
            mark = "ok  " if entry["agrees"] else "DIFF"
            print(f"  [{mark}] {entry['id']}: {entry['detail']}")

        agreed = sum(1 for entry in entries if entry["agrees"])
        report = {
            "dataset": str(config.dataset),
            "sql_model": ollama_config.sql_model,
            "checked": len(entries),
            "agreed": agreed,
            "disagreed": len(entries) - agreed,
            "entries": entries,
        }
        write_json(config.report, report)

        print(f"\n=== SQL VERIFICATION ===")
        print(
            f"checked={len(entries)} agreed={agreed} disagreed={len(entries) - agreed}"
        )
        print(f"Report written: {config.report}")
        if agreed != len(entries):
            print(
                "A disagreement means the two derivations differ, not that the gold is "
                "wrong. Read the recorded SQL before changing any answer."
            )
        return 0 if agreed == len(entries) else 1
    finally:
        helper_postgres.close_connection()
