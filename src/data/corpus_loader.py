"""Corpus loading and hashing.

Every generator computes line numbers and evidence hashes through these
functions, and ``validate.py`` re-derives them the same way at verification time.
That shared path is the point: a hash computed one way at generation and another
way at validation would make every record fail, or worse, make every record pass
for the wrong reason.

Split assignment deliberately does not live here. It is not a per-line or
per-group property but a property of the whole record set, and it is resolved in
``src.utils.helper_splits`` once every record exists.
"""

import hashlib
from pathlib import Path


def lines_from_bytes(data: bytes) -> list[str]:
    """Splits corpus bytes into lines, dropping a single trailing newline.

    Decoding is lossy on purpose. LogHub files carry raw bytes from real systems,
    and a decode error must not be able to stop a run over a file that already
    checksum-verified against the lock.

    Args:
        data: Raw bytes of a ``*_2k.log`` file.

    Returns:
        The file's lines, without terminators. Index ``i`` is line ``i + 1``.
    """
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def load_lines(log_path: Path) -> list[str]:
    """Reads a corpus file into its lines.

    Args:
        log_path: Path to a ``*_2k.log`` file.

    Returns:
        The file's lines, without terminators. Index ``i`` is line ``i + 1``.
    """
    return lines_from_bytes(log_path.read_bytes())


def sha256_line(line: str) -> str:
    """Returns the ``sha256:``-prefixed digest of one line's UTF-8 bytes.

    Args:
        line: The line's text, without its terminator.

    Returns:
        The prefixed hex digest.
    """
    return "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Returns the ``sha256:``-prefixed digest of a byte string.

    Args:
        data: Bytes to hash.

    Returns:
        The prefixed hex digest.
    """
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Returns the ``sha256:``-prefixed digest of a file's contents.

    Used to record which corpus, schema and input files a validation report was
    produced from, so a report can be tied to the artefacts it certified rather
    than only to a path that may since have been rewritten.

    Args:
        path: File to hash.

    Returns:
        The prefixed hex digest.
    """
    return sha256_bytes(path.read_bytes())


def dataset_key(name: str) -> str:
    """Returns the lowercase dataset name used as the evidence-id prefix.

    This is also the value in Postgres's ``lines.dataset`` column, which is what
    lets ``src.utils.helper_postgres`` resolve a record's evidence back to real
    rows from the ``id`` string alone.

    Args:
        name: Dataset name as declared in ``dataset_specs`` (e.g. ``"HDFS"``).

    Returns:
        The lowercase key (e.g. ``"hdfs"``).
    """
    return name.lower()
