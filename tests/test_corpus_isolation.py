"""Asserts the corpus fetcher stays importable inside the loghub container.

``src/corpus/fetch_corpus.py`` runs in a Postgres image that has Python and
``psycopg2`` and nothing else. If it ever imports from the rest of ``src`` — even
transitively, even something as small as a hashing helper — the loghub container
starts failing at startup on a missing ``jsonschema``, and the failure surfaces as
a corpus that never loads rather than as an import error anyone would read.

The boundary is cheap to state and easy to break by habit, so it is asserted here.
"""

import ast
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FETCHER = REPO_ROOT / "src" / "corpus" / "fetch_corpus.py"
ALLOWED_THIRD_PARTY = {"psycopg2"}


def imported_modules(path: Path) -> set[str]:
    """Collects the top-level module names a file imports.

    Args:
        path: Python file to parse.

    Returns:
        Top-level names from every ``import`` and ``from ... import``, including
        those inside functions, since a deferred import is still a dependency.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


class CorpusIsolationTest(unittest.TestCase):
    def test_fetcher_exists_where_the_dockerfile_expects_it(self):
        self.assertTrue(FETCHER.is_file(), f"{FETCHER} is missing")
        self.assertTrue(
            (FETCHER.parent / "corpus_manifest.json").is_file(),
            "the manifest must sit beside the fetcher that honours it",
        )

    def test_fetcher_does_not_import_the_application(self):
        imports = imported_modules(FETCHER)
        self.assertNotIn("src", imports)
        self.assertNotIn("config", imports)

    def test_fetcher_third_party_imports_are_available_in_the_loghub_image(self):
        imports = imported_modules(FETCHER)
        standard_library = set(sys.stdlib_module_names)
        third_party = imports - standard_library - {"src", "config"}
        self.assertTrue(
            third_party <= ALLOWED_THIRD_PARTY,
            f"the loghub image installs only {sorted(ALLOWED_THIRD_PARTY)}; "
            f"found {sorted(third_party - ALLOWED_THIRD_PARTY)}",
        )

    def test_dockerfile_copies_the_moved_paths(self):
        dockerfile = (REPO_ROOT / "docker" / "loghub" / "Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("src/corpus/fetch_corpus.py", dockerfile)
        self.assertIn("src/corpus/corpus_manifest.json", dockerfile)

    def test_dockerignore_does_not_exclude_the_corpus_package(self):
        ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        for line in ignored.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            self.assertNotIn(
                "loghub",
                stripped,
                "a stale loghub/ exclusion would keep the fetcher out of the "
                "build context",
            )


if __name__ == "__main__":
    unittest.main()
