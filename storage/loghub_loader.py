"""Loghub-2.0 data loader.

Downloads the pre-parsed structured CSV files that Loghub-2.0
(https://github.com/logpai/loghub-2.0) ships for each of its systems under
the ``2k_dataset`` directory, and converts each row into a normalized
dictionary ready for storage.

Reference:
    LogRouter paper, Section IV-A (Datasets and Question Sets).
"""

import csv
import io
import json
from typing import Any

import requests

LOGHUB_RAW_BASE = "https://raw.githubusercontent.com/logpai/loghub-2.0/main/2k_dataset"

KNOWN_DATASETS = [
    "Hadoop",
    "HDFS",
    "OpenStack",
    "Spark",
    "Zookeeper",
    "BGL",
    "HPC",
    "Thunderbird",
    "Linux",
    "Mac",
    "Apache",
    "OpenSSH",
    "HealthApp",
    "Proxifier",
]

COMMON_COLUMNS = {"LineId", "Content", "EventId", "EventTemplate", "ParameterList"}


def fetch_structured_csv(dataset: str, timeout: int = 60) -> list[dict[str, str]]:
    """Downloads and parses a Loghub-2.0 structured CSV file.

    Args:
        dataset: Name of the Loghub-2.0 system, for example ``"HDFS"``.
        timeout: Timeout in seconds for the HTTP request.

    Returns:
        list[dict[str, str]]: One dictionary per log line, with the columns
        defined by Loghub-2.0's annotation format (``LineId``, ``Content``,
        ``EventId``, ``EventTemplate``, ``ParameterList``, and any
        system-specific columns).

    Raises:
        requests.HTTPError: If the CSV file cannot be downloaded.
    """
    url = f"{LOGHUB_RAW_BASE}/{dataset}/{dataset}_2k.log_structured.csv"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    return list(reader)


def to_row(dataset: str, raw_row: dict[str, str]) -> dict[str, Any]:
    """Converts a raw Loghub-2.0 CSV row into a normalized storage row.

    Fields shared across every Loghub-2.0 system are promoted to typed
    columns; every other, system-specific column is kept as JSON in
    ``raw_fields`` so that a single schema can hold all systems.

    Args:
        dataset: Name of the Loghub-2.0 system this row belongs to.
        raw_row: A single row as returned by :func:`fetch_structured_csv`.

    Returns:
        dict[str, Any]: A row with the keys ``dataset``, ``line_id``,
        ``content``, ``event_id``, ``event_template``, ``parameter_list``,
        and ``raw_fields``.
    """
    raw_fields = {key: value for key, value in raw_row.items() if key not in COMMON_COLUMNS}
    return {
        "dataset": dataset,
        "line_id": int(raw_row["LineId"]),
        "content": raw_row.get("Content"),
        "event_id": raw_row.get("EventId"),
        "event_template": raw_row.get("EventTemplate"),
        "parameter_list": raw_row.get("ParameterList"),
        "raw_fields": json.dumps(raw_fields, ensure_ascii=False),
    }


def load_dataset(dataset: str) -> list[dict[str, Any]]:
    """Fetches and normalizes every row of one Loghub-2.0 system.

    Args:
        dataset: Name of the Loghub-2.0 system, for example ``"HDFS"``.

    Returns:
        list[dict[str, Any]]: Normalized rows, as produced by
        :func:`to_row`, ready to be inserted into the ``logs`` table.
    """
    raw_rows = fetch_structured_csv(dataset)
    return [to_row(dataset, raw_row) for raw_row in raw_rows]
