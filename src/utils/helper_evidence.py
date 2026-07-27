"""Record-building primitives shared by the three question tiers.

Everything here shapes a field that ``config/question_schema.json`` constrains,
which is why it is shared rather than repeated per tier: an evidence id built one
way in the easy tier and another way in the hard tier would still validate
against the schema and still be wrong, because the id is what
``validate.py::dataset_key_from_evidence`` resolves a record back to its corpus
with.
"""

import re
from typing import Any

from src.data.corpus_loader import sha256_line


def slugify(text: str) -> str:
    """Reduces free text to the ``[a-z0-9_]`` alphabet record ids are limited to.

    ``config/question_schema.json`` pins ``id`` to ``^[a-z0-9_]+$``, so a literal
    that reaches an id — and literals routinely contain spaces, brackets and
    colons — has to pass through here first.

    Args:
        text: Arbitrary text, typically a search literal or a group key.

    Returns:
        The slugified form, with runs of other characters collapsed to ``_``.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def evidence_ref(
    dataset_key: str, line_number: int, line_text: str, group_id: str
) -> dict[str, Any]:
    """Builds one ``evidence.refs`` entry.

    The id embeds the dataset, the zero-padded line number and a slice of the
    line hash. That last part is what makes the reference self-checking: the
    validator re-reads line ``line_number`` from Postgres and re-hashes it, so a
    corpus that changed under a record cannot go unnoticed.

    Args:
        dataset_key: Lowercase dataset key (Postgres ``lines.dataset``).
        line_number: 1-based line number.
        line_text: The line's text, hashed into ``line_hash``.
        group_id: The evidence group this line belongs to; every question sharing
            it lands in the same dev/test split.

    Returns:
        An ``evidence.refs`` entry.
    """
    line_hash = sha256_line(line_text)
    return {
        "id": f"{dataset_key}:line:{line_number:08d}:{line_hash[7:23]}",
        "line_number": line_number,
        "line_hash": line_hash,
        "group_id": group_id,
    }


def gold_provenance(
    method: str,
    created_by: str,
    created_at: str,
    corpus_sha256: str,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds a record's ``gold_provenance`` block.

    The ``model`` block is included only when one was actually used. The schema
    enforces this in both directions — required for
    ``independent_model_then_human``, forbidden for
    ``deterministic_aggregation`` — so an empty model block on a deterministic
    record is not a harmless extra field but a claim that a model was involved.

    Args:
        method: ``deterministic_aggregation``, ``independent_model_then_human``
            or ``human``.
        created_by: The module and version that produced the record.
        created_at: Fixed timestamp; see ``GenerationConfig.created_at``.
        corpus_sha256: Whole-file digest of the corpus the record was built from.
        model: The drafting model's provenance block, or ``None``.

    Returns:
        A ``gold_provenance`` block.
    """
    provenance = {
        "method": method,
        "created_by": created_by,
        "created_at": created_at,
        "corpus_sha256": corpus_sha256,
    }
    if model is not None:
        provenance["model"] = model
    return provenance


def split_sentences(text: str) -> list[str]:
    """Splits an answer into sentences.

    Used by the hard tier to turn one drafted answer into the individual claims
    the groundedness model checks one at a time — a whole-answer verdict would
    hide which part of a four-sentence synthesis was unsupported.

    Args:
        text: The drafted answer.

    Returns:
        Non-empty, stripped sentences.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
