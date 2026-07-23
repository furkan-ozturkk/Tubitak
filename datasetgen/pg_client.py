#!/usr/bin/env python3
"""
pg_client.py

Queries loghub's Postgres (+pgvector) corpus store (the `lines` table) over
datasetgen-net, instead of reading the raw *_2k.log files directly. Used by
question_generators.py's easy tier (_count_matches) and schema_validator.py's
numeric_claims / evidence-hash checks.

Connection is read from the same env vars docker-compose.yml sets on the
datasetgen service: PGHOST, PGPORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD.
"""
import os

import psycopg2

_connection = None


def _connect():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "loghub"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "loghub"),
        user=os.environ.get("POSTGRES_USER", "loghub"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def get_connection():
    global _connection
    if _connection is None or _connection.closed:
        _connection = _connect()
    return _connection


def count_literal(dataset: str, literal: str, case_sensitive: bool = False):
    """Mirrors the old in-memory _count_matches semantics exactly:
    case-insensitive (unless case_sensitive) substring containment, counted
    per line. Returns (count, matched_line_numbers) with 1-based line
    numbers, ascending -- same contract the old version returned."""
    conn = get_connection()
    with conn.cursor() as cur:
        if case_sensitive:
            cur.execute(
                "SELECT line_number FROM lines WHERE dataset = %s AND strpos(text, %s) > 0 "
                "ORDER BY line_number",
                (dataset, literal),
            )
        else:
            cur.execute(
                "SELECT line_number FROM lines WHERE dataset = %s "
                "AND strpos(lower(text), lower(%s)) > 0 ORDER BY line_number",
                (dataset, literal),
            )
        rows = [r[0] for r in cur.fetchall()]
    return len(rows), rows


def count_regex(dataset: str, pattern: str, case_sensitive: bool = False):
    """Postgres regex match (~ / ~*). Returns (count, matched_line_numbers)."""
    conn = get_connection()
    op = "~" if case_sensitive else "~*"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT line_number FROM lines WHERE dataset = %s AND text {op} %s ORDER BY line_number",
            (dataset, pattern),
        )
        rows = [r[0] for r in cur.fetchall()]
    return len(rows), rows


def get_line(dataset: str, line_number: int) -> str:
    """Fetch a single line's text by (dataset, line_number)."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT text FROM lines WHERE dataset = %s AND line_number = %s",
            (dataset, line_number),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"pg_client.get_line: no row for dataset={dataset} line_number={line_number}")
    return row[0]
