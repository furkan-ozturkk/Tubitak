"""Generates structured queries for the SQL path.

Combines a template match with a coder LLM to produce a Postgres query (in
place of the paper's Druid SQL); the result is returned directly without
further LLM rewriting.

Reference:
    LogRouter paper, Section III-C (SQL Path).
"""
