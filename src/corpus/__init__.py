"""The pinned corpus source and the fetcher that honours it.

Deliberately a leaf. This package runs inside the loghub container, which is a
Postgres image with Python added for one script, so nothing here imports from the
rest of ``src``: pulling the question-generation app into that image would give it
a dependency set it has no use for and cannot install.
``tests/test_corpus_isolation.py`` asserts that boundary instead of trusting it.

``corpus_manifest.json`` pins the commit and the ten datasets.
``fetch_corpus.py`` fetches them, verifies their digests against a lock, and loads
them into the ``lines`` table the rest of the project queries.

The manifest sits beside the fetcher rather than in ``config/`` because the two are
one contract: the pin is only meaningful together with the code that refuses to
proceed when a digest disagrees with it.
"""
