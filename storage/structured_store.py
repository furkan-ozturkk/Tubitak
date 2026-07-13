"""Keyword and SQL storage backend.

The paper uses Apache Druid; at this project's scale the same role is served
by a Postgres table (``logs``) with full-text search.

Reference:
    LogRouter paper, Section III-A (Storage and orchestration).
"""
