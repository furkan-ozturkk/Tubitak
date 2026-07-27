"""Asserts every Python source in the repo explains itself in docstrings, not comments.

The repo's convention is that Python carries no ``#`` comment lines: an explanation
belongs in a module, class or function docstring, where it is addressable, reachable
from ``help()``, and cannot drift to the wrong side of a refactor. Shell, YAML,
Dockerfiles and SQL keep their comments, because those formats have nowhere else to put
prose.

Shebang lines, encoding declarations and ``# type:`` directives are instructions to a
tool rather than prose, and are allowed. A ``#`` inside a string literal is not a
comment, which is why this tokenises the source instead of pattern-matching lines.

Usage:
  python3 scripts/check_no_comments.py
  python3 scripts/check_no_comments.py src/ validate.py
"""

import argparse
import io
import os
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRECTORIES = {".git", ".venv", "__pycache__", "GNN-Backdoor", "node_modules"}
ALLOWED_PREFIXES = ("#!", "# -*-", "# type:")


def python_files(targets: list[Path]) -> list[Path]:
    """Collects the Python files to check.

    Args:
        targets: Files or directories to walk.

    Returns:
        Python file paths, sorted and deduplicated, excluding vendored and generated
        directories.
    """
    found: list[Path] = []
    for target in targets:
        if target.is_file():
            if target.suffix == ".py":
                found.append(target)
            continue
        for root, directories, filenames in os.walk(target):
            directories[:] = [
                name for name in directories if name not in SKIP_DIRECTORIES
            ]
            for filename in filenames:
                if filename.endswith(".py"):
                    found.append(Path(root) / filename)
    return sorted(set(found))


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


def main() -> int:
    """Checks the requested paths and reports every comment found.

    Returns:
        ``0`` when no Python file carries a comment, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        default=None,
        help="Files or directories to check; defaults to the whole repository",
    )
    args = parser.parse_args()

    targets = [Path(target) for target in args.targets] if args.targets else [REPO_ROOT]
    total = 0
    for path in python_files(targets):
        findings = comments_in(path)
        if not findings:
            continue
        relative = path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path
        for line_number, text in findings:
            preview = text if len(text) <= 70 else text[:67] + "..."
            print(f"{relative}:{line_number}: {preview}", file=sys.stderr)
            total += 1

    if total:
        print(
            f"\n{total} comment(s) found. Move the explanation into a docstring.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
