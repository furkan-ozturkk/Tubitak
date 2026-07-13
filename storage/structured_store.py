"""Keyword and SQL storage backend.

The paper uses Apache Druid; at this project's scale the same role is
served by a Postgres table (``logs``) with full-text search. This module
owns the connection, schema, and read/write access to that table.

Reference:
    LogRouter paper, Section III-A (Storage and orchestration).
"""

import time
from typing import Any, Optional

import psycopg2
import psycopg2.extras

from config import Config

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS logs (
    id              BIGSERIAL PRIMARY KEY,
    dataset         TEXT NOT NULL,
    line_id         INTEGER NOT NULL,
    content         TEXT,
    event_id        TEXT,
    event_template  TEXT,
    parameter_list  TEXT,
    raw_fields      JSONB,
    inserted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dataset, line_id)
);
CREATE INDEX IF NOT EXISTS idx_logs_dataset ON logs (dataset);
CREATE INDEX IF NOT EXISTS idx_logs_event_id ON logs (dataset, event_id);
"""

INSERT_SQL = """
INSERT INTO logs (dataset, line_id, content, event_id, event_template, parameter_list, raw_fields)
VALUES %s
ON CONFLICT (dataset, line_id) DO NOTHING
"""


def connect(config: Config, retries: int = 20, delay: float = 3.0) -> "psycopg2.extensions.connection":
    """Opens a Postgres connection, retrying while the database starts up.

    Args:
        config: Configuration holding the Postgres connection settings.
        retries: Maximum number of connection attempts.
        delay: Delay in seconds between attempts.

    Returns:
        psycopg2.extensions.connection: An open connection to Postgres.

    Raises:
        psycopg2.OperationalError: If no connection could be established
            after ``retries`` attempts.
    """
    dsn = (
        f"host={config.postgres_host} port={config.postgres_port} "
        f"user={config.postgres_user} password={config.postgres_password} "
        f"dbname={config.postgres_db}"
    )
    last_error: Optional[Exception] = None
    for _ in range(retries):
        try:
            return psycopg2.connect(dsn)
        except psycopg2.OperationalError as error:
            last_error = error
            time.sleep(delay)
    raise last_error


def ensure_schema(conn: "psycopg2.extensions.connection") -> None:
    """Creates the ``logs`` table and its indexes if they do not exist yet.

    Args:
        conn: An open Postgres connection.
    """
    with conn.cursor() as cursor:
        cursor.execute(CREATE_TABLE_SQL)
    conn.commit()


def insert_rows(conn: "psycopg2.extensions.connection", rows: list[dict[str, Any]]) -> int:
    """Inserts normalized log rows into the ``logs`` table.

    Existing rows with the same ``(dataset, line_id)`` pair are left
    untouched, so this function is safe to call multiple times with the
    same data.

    Args:
        conn: An open Postgres connection.
        rows: Normalized rows as produced by
            :func:`storage.loghub_loader.to_row`.

    Returns:
        int: The number of rows passed to the insert statement.
    """
    values = [
        (
            row["dataset"],
            row["line_id"],
            row["content"],
            row["event_id"],
            row["event_template"],
            row["parameter_list"],
            row["raw_fields"],
        )
        for row in rows
    ]
    with conn.cursor() as cursor:
        psycopg2.extras.execute_values(cursor, INSERT_SQL, values)
    conn.commit()
    return len(values)


def fetch_rows(
    conn: "psycopg2.extensions.connection", dataset: str, limit: int = 100
) -> list[dict[str, Any]]:
    """Reads back rows for one dataset from the ``logs`` table.

    Args:
        conn: An open Postgres connection.
        dataset: Name of the Loghub-2.0 system to filter on.
        limit: Maximum number of rows to return.

    Returns:
        list[dict[str, Any]]: Matching rows as dictionaries.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute(
            "SELECT * FROM logs WHERE dataset = %s ORDER BY line_id LIMIT %s",
            (dataset, limit),
        )
        return [dict(row) for row in cursor.fetchall()]
