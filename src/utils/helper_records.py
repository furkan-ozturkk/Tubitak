"""Reading question records back off disk.

``validate.py`` and ``analysis/`` both start from the same place — a written
dataset file — so both load it through here. Records are accepted as a JSON
array or as JSONL, because the pilot output is an array and a scaled run
(Section 3.2) streams.

Every loaded record carries a ``_source_file`` key that is *not* part of the
schema. ``validate.py`` strips it before validating and reports it in error
locations, which is what lets a failure across a multi-file glob name the file
it came from.
"""

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SOURCE_FILE_KEY = "_source_file"


def load_questions(patterns: Iterable[str]) -> list[dict[str, Any]]:
    """Loads question records from files or glob patterns.

    Args:
        patterns: File paths or glob patterns. ``.jsonl`` files are read one
            record per line; anything else is read as a JSON array (a bare
            object is accepted as a one-record file).

    Returns:
        Every record found, each tagged with its source file.

    Raises:
        SystemExit: If a JSONL line is not valid JSON. The line number is
            included, because a truncated streamed write is the likely cause and
            it is worth knowing where it stopped.
    """
    records: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(glob.glob(str(pattern))):
            p = Path(path)
            if p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise SystemExit(f"ERROR: {p}:{lineno} invalid JSON: {e}")
                        rec[SOURCE_FILE_KEY] = str(p)
                        records.append(rec)
            else:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for rec in items:
                    rec[SOURCE_FILE_KEY] = str(p)
                    records.append(rec)
    return records


def strip_source_file(record: dict[str, Any]) -> dict[str, Any]:
    """Returns the record without the loader's ``_source_file`` annotation.

    Args:
        record: A loaded record.

    Returns:
        A copy carrying only schema fields.
    """
    return {k: v for k, v in record.items() if k != SOURCE_FILE_KEY}


def dataset_key_from_evidence(record: dict[str, Any]) -> str | None:
    """Recovers which LogHub dataset a record was built from.

    Read off the first evidence id rather than stored as its own field: the
    dataset a record belongs to is a property of the lines it cites, and a
    separate field could disagree with them.

    Args:
        record: A question record.

    Returns:
        The lowercase dataset key, or ``None`` when the record cites nothing.
    """
    refs = record.get("evidence", {}).get("refs", [])
    if not refs:
        return None
    ref_id = refs[0].get("id", "")
    return ref_id.split(":", 1)[0] if ":" in ref_id else None


def count_by(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    """Counts records per value of one field.

    Args:
        records: Question records.
        field: Field to group by.

    Returns:
        Value-to-count mapping; records missing the field count under ``"?"``.
    """
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.get(field, "?")] += 1
    return dict(counts)
