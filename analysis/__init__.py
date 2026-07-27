"""Offline analysis of a finished dataset.

Nothing here generates or validates anything. Every module reads an
``output/**/questions.json`` (and, where present, the ``validation_report.json``
beside it), so the analysis re-runs against a finished dataset without a corpus
volume, a database, or an Ollama server.

Flat by design, and each module is an entry point with its own ``argparse`` — the
project's single CLI surface in ``config/args.py`` is for the pipeline that
*produces* the dataset, and a reporting tool that borrowed it would inherit
sixty flags it has no use for.

``analysis_tables`` – dataset composition tables: per tier, per routing path, per
                      split, per review status, and per LogHub dataset.

The distinction that matters when reading a table: ``review_status=verified`` on
an easy-tier row means a machine recomputed the answer, whereas on a medium or
hard row it means a human accepted a model's draft. Both are verified; they are
not the same claim, and the per-tier breakdown is what keeps them apart.
"""
