"""Asserts every Python source explains itself in docstrings, not comments.

This is a test rather than a script because it is a property the codebase must
hold, not an operator tool: it belongs with the other assertions, where
``unittest discover`` runs it without anyone remembering to.

The convention: an explanation goes in a module, class or function docstring,
where it is addressable, reachable from ``help()``, and cannot drift to the wrong
side of a refactor. Shell, YAML, Dockerfiles and SQL keep their comments, having
nowhere else to put prose.

Shebangs, encoding declarations and ``# type:`` directives are instructions to a
tool rather than prose, and are allowed. A ``#`` inside a string literal is not a
comment, which is why this tokenises the source instead of matching lines.
"""

import io
import os
import tokenize
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRECTORIES = {".git", ".venv", "__pycache__", "GNN-Backdoor", "node_modules"}
ALLOWED_PREFIXES = ("#!", "# -*-", "# type:")


def python_files() -> list[Path]:
    """Collects every Python file in the repository.

    Returns:
        Python file paths, sorted, excluding vendored and generated directories.
    """
    found: list[Path] = []
    for root, directories, filenames in os.walk(REPO_ROOT):
        directories[:] = [name for name in directories if name not in SKIP_DIRECTORIES]
        for filename in filenames:
            if filename.endswith(".py"):
                found.append(Path(root) / filename)
    return sorted(found)


def comments_in(path: Path) -> list[tuple[int, str]]:
    """Finds the disallowed comment tokens in one file.

    Args:
        path: Python file to tokenise.

    Returns:
        ``(line_number, text)`` pairs for every comment that is not an allowed
        directive. A file that cannot be tokenised is reported as a finding rather
        than skipped, since silently passing an unparseable file would make this
        check weaker the more broken the code is.
    """
    source = path.read_text(encoding="utf-8")
    findings: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            text = token.string.strip()
            if text.startswith(ALLOWED_PREFIXES):
                continue
            findings.append((token.start[0], text))
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        findings.append((0, f"could not tokenise: {error}"))
    return findings


class NoCommentsTest(unittest.TestCase):
    def test_repository_finds_python_files(self):
        self.assertGreater(len(python_files()), 20)

    def test_no_python_source_carries_a_comment(self):
        offences: list[str] = []
        for path in python_files():
            for line_number, text in comments_in(path):
                preview = text if len(text) <= 70 else text[:67] + "..."
                offences.append(
                    f"{path.relative_to(REPO_ROOT)}:{line_number}: {preview}"
                )
        self.assertEqual(
            offences,
            [],
            "move these explanations into docstrings:\n  " + "\n  ".join(offences),
        )


if __name__ == "__main__":
    unittest.main()
