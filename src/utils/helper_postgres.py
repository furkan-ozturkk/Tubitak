"""Client for loghub's Postgres (+pgvector) corpus store.

A client, and so a ``helper_*`` beside ``helper_vllm``, not a member of
``src.data``: that package models the corpus itself — its bytes, its hashing — and
this is the transport that happens to reach a database over ``datasetgen-net``.
The two systems this project talks to sit next to each other under one
convention.

The ``lines`` table, not the raw ``*_2k.log`` files, answers every question whose
gold value must be reproducible: the easy tier's counts and lookups
(``src.generators.easy_tier``) and ``validate.py``'s evidence-hash and
``numeric_claims`` re-derivation. Both sides issue the same SQL against the same
table, so a gold count and its verification cannot disagree because one of them
scanned a file differently.

That shared table is also the limit of what this proves, and ``validate.py``'s
``check_corpus_matches_database`` exists because of it: if the loader dropped,
duplicated or misnumbered a line, generation and validation would agree on the
same wrong table. The database is checked against the raw file it was loaded from
before any answer computed from it is believed.

``psycopg2`` is imported inside ``_connect`` rather than at module scope so that
``main.py --help``, the CLI validations and every offline analysis keep working on
a machine with no database driver installed.

Connection is read from the env vars ``docker/compose.yml`` sets on the datasetgen
service: ``PGHOST``, ``PGPORT``, ``POSTGRES_DB``, ``POSTGRES_USER``,
``POSTGRES_PASSWORD``.
"""

import os
import re
from typing import Any

CONNECT_TIMEOUT_SECONDS = 10
STATEMENT_TIMEOUT_MS = 30_000

_connection: Any = None


def _connect() -> Any:
    """Opens a new connection with both timeouts set.

    ``connect_timeout`` bounds the handshake, so a container that starts before
    Postgres is reachable fails in seconds instead of hanging. ``statement_timeout``
    bounds each query, which matters because ``count_regex`` runs a pattern taken
    from a record: a pathological regex would otherwise pin a backend
    indefinitely.

    Returns:
        A new psycopg2 connection.
    """
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("PGHOST", "loghub"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "loghub"),
        user=os.environ.get("POSTGRES_USER", "loghub"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}",
    )


def get_connection() -> Any:
    """Returns the process-wide connection, reconnecting if it has dropped.

    Returns:
        An open connection.
    """
    global _connection
    if _connection is None or _connection.closed:
        _connection = _connect()
    return _connection


def close_connection() -> None:
    """Closes the process-wide connection if one is open.

    Called at the end of a command so a long-lived container does not accumulate
    idle backends across repeated ``docker compose exec`` invocations.
    """
    global _connection
    if _connection is not None and not _connection.closed:
        _connection.close()
    _connection = None


def _fetch_line_numbers(sql: str, params: tuple) -> list[int]:
    """Runs a query returning line numbers, rolling back on failure.

    Every read here runs inside psycopg2's implicit transaction. Without the
    rollback, one failed statement — an invalid user-supplied regex is the
    realistic case — leaves the connection in an aborted transaction and every
    later query fails with a misleading error about the transaction rather than
    about the pattern that broke it.

    Args:
        sql: Query returning a single ``line_number`` column.
        params: Query parameters.

    Returns:
        Line numbers, ascending.

    Raises:
        Exception: Whatever the driver raised, after the rollback.
    """
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = [row[0] for row in cursor.fetchall()]
    except Exception:
        connection.rollback()
        raise
    connection.commit()
    return rows


def count_literal(
    dataset: str, literal: str, case_sensitive: bool = False
) -> tuple[int, list[int]]:
    """Counts the lines of one dataset containing a literal.

    Substring containment, case-insensitive unless ``case_sensitive``. The matched
    line numbers are returned alongside the count because the easy tier cites the
    first few of them as evidence, and a second query to find them could disagree
    with the count if the table changed in between.

    Args:
        dataset: Lowercase dataset key.
        literal: Substring to match.
        case_sensitive: Whether matching is case sensitive.

    Returns:
        Tuple ``(count, line_numbers)`` with 1-based line numbers, ascending.
    """
    if case_sensitive:
        sql = (
            "SELECT line_number FROM lines WHERE dataset = %s AND strpos(text, %s) > 0 "
            "ORDER BY line_number"
        )
    else:
        sql = (
            "SELECT line_number FROM lines WHERE dataset = %s "
            "AND strpos(lower(text), lower(%s)) > 0 ORDER BY line_number"
        )
    rows = _fetch_line_numbers(sql, (dataset, literal))
    return len(rows), rows


def count_regex(
    dataset: str, pattern: str, case_sensitive: bool = False
) -> tuple[int, list[int]]:
    """Counts the lines of one dataset matching a regular expression.

    Args:
        dataset: Lowercase dataset key.
        pattern: POSIX regular expression, evaluated by Postgres.
        case_sensitive: Selects the ``~`` or ``~*`` operator.

    Returns:
        Tuple ``(count, line_numbers)`` with 1-based line numbers, ascending.

    Raises:
        Exception: If Postgres rejects the pattern, or the statement timeout is
            hit. The caller reports it as a record-level error.
    """
    operator = "~" if case_sensitive else "~*"
    sql = (
        f"SELECT line_number FROM lines WHERE dataset = %s AND text {operator} %s "
        f"ORDER BY line_number"
    )
    rows = _fetch_line_numbers(sql, (dataset, pattern))
    return len(rows), rows


def get_line(dataset: str, line_number: int) -> str:
    """Fetches one line's text by dataset and line number.

    Args:
        dataset: Lowercase dataset key.
        line_number: 1-based line number.

    Returns:
        The line's text.

    Raises:
        RuntimeError: If no such row exists, which means the record cites a line
            the loaded corpus does not have.
    """
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT text FROM lines WHERE dataset = %s AND line_number = %s",
                (dataset, line_number),
            )
            row = cursor.fetchone()
    except Exception:
        connection.rollback()
        raise
    connection.commit()
    if row is None:
        raise RuntimeError(
            f"helper_postgres.get_line: no row for dataset={dataset} "
            f"line_number={line_number}"
        )
    return row[0]


FORBIDDEN_SQL_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "copy",
    "call",
    "vacuum",
    "reindex",
    "set",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_sleep",
    "dblink",
    "lo_import",
    "lo_export",
)

_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_SQL_KEYWORDS) + r")\b", re.IGNORECASE
)


def assert_readonly_select(sql: str) -> None:
    """Rejects anything that is not a single read-only query.

    Model-written SQL is untrusted input, so this is the gate in front of it. Two
    independent defences, because either alone is thin: this keyword and shape
    check, and the READ ONLY transaction ``run_readonly_query`` opens, which
    Postgres itself enforces. A keyword list can be evaded; a read-only transaction
    cannot, and a read-only transaction still permits a thousand-second scan, which
    is what the statement timeout is for.

    Matching is on word boundaries, not substrings. A substring check reads ``set``
    inside ``dataset = 'linux'`` and refuses the most ordinary query this project
    issues, which is how a guard ends up switched off for being too noisy.

    Args:
        sql: The candidate query.

    Raises:
        ValueError: If the statement is not a lone SELECT or WITH, or mentions a
            writing or filesystem construct.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("empty query")
    if ";" in stripped:
        raise ValueError("multiple statements are not allowed")
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("only a single SELECT (or WITH ... SELECT) is allowed")
    found = _FORBIDDEN_SQL_PATTERN.search(stripped)
    if found:
        raise ValueError(f"forbidden construct in query: {found.group(1).lower()!r}")


def run_readonly_query(sql: str) -> list[tuple]:
    """Runs an untrusted query inside a read-only transaction.

    Args:
        sql: The query, already passed through ``assert_readonly_select``.

    Returns:
        Every returned row.

    Raises:
        ValueError: If the query fails the read-only shape check.
        Exception: Whatever the driver raised, after the rollback.
    """
    assert_readonly_select(sql)
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(sql)
            rows = cursor.fetchall()
    except Exception:
        connection.rollback()
        raise
    connection.rollback()
    return rows


def fetch_all_lines(dataset: str) -> list[tuple[int, str]]:
    """Fetches every line of one dataset, ordered by line number.

    Used by ``validate.py`` to prove the loaded table equals the raw file, which is
    the one claim no per-line lookup can establish: a lookup only shows that the
    lines a record happens to cite are intact.

    Args:
        dataset: Lowercase dataset key.

    Returns:
        ``(line_number, text)`` pairs, ascending.
    """
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT line_number, text FROM lines WHERE dataset = %s ORDER BY line_number",
                (dataset,),
            )
            rows = [(row[0], row[1]) for row in cursor.fetchall()]
    except Exception:
        connection.rollback()
        raise
    connection.commit()
    return rows
