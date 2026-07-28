"""Asserts the one line contract every hash and line number depends on.

Two implementations split corpus bytes into lines: ``src.data.corpus_loader``
(the datasetgen side) and ``src.corpus.fetch_corpus`` (the loghub container,
which may not import from ``src`` and therefore mirrors the logic). If they
ever disagree, generation and the loaded ``lines`` table describe different
corpora while every checksum still passes — the exact failure
``validate.py::check_corpus_matches_database`` exists to catch, except it
would then be *systematic* rather than accidental.

The contract also has an external consumer: the LLM Log Analyzer evaluation
harness reads the same files with ``str.splitlines()`` and recomputes every
``line_hash`` and evidence ``id``. The golden vectors below pin the CRLF case
(the LogHub 2k files use CRLF endings) so a regression back to a ``\\n``-only
split — which keeps a ``\\r`` in every line and silently changes every hash —
fails here first.
"""

import hashlib
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.corpus.fetch_corpus import corpus_lines
from src.data.corpus_loader import lines_from_bytes, sha256_line

VECTORS = (
    b"",
    b"a",
    b"a\n",
    b"a\nb",
    b"a\nb\n",
    b"a\nb\n\n",
    b"a\r\nb\r\n",
    b"a\r\nb",
    b"mixed\nunix\r\nwindows\rmac\n",
    b"trailing space \r\n",
    b"a\xffb\r\n",
)

CRLF_SAMPLE = (
    b"Jun 14 15:16:01 combo sshd(pam_unix)[19939]: check pass; user unknown\r\n"
)


class LineContractTest(unittest.TestCase):
    def test_both_implementations_agree_on_every_vector(self):
        for data in VECTORS:
            self.assertEqual(
                lines_from_bytes(data), corpus_lines(data), f"vector {data!r}"
            )

    def test_crlf_terminator_never_reaches_a_line(self):
        for line in lines_from_bytes(b"a\r\nb\r\nc"):
            self.assertNotIn("\r", line)

    def test_crlf_and_lf_files_hash_identically(self):
        crlf = lines_from_bytes(b"one\r\ntwo\r\n")
        lf = lines_from_bytes(b"one\ntwo\n")
        self.assertEqual(
            [sha256_line(line) for line in crlf],
            [sha256_line(line) for line in lf],
        )

    def test_golden_crlf_line_hash(self):
        (line,) = lines_from_bytes(CRLF_SAMPLE)
        expected = "sha256:" + hashlib.sha256(CRLF_SAMPLE[:-2]).hexdigest()
        self.assertEqual(sha256_line(line), expected)

    def test_splitlines_semantics_not_newline_split(self):
        self.assertEqual(lines_from_bytes(b"a\rb"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
