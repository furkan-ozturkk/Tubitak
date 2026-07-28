"""The pipeline's command stages, one module per ``--command``.

``main.py`` is the only executable entry point; each module here is the library
implementation of one command it dispatches, named for what the stage does
rather than for the flag that invokes it:

``generate``        – ``--command generate``: the three question tiers.
``validate``        – ``--command validate``: schema, cross-record, split,
                      corpus and answer checks.
``sql_verification``– ``--command verify-answers``: an independent model writes
                      its own SQL per question; its result is compared to gold.
``analyzer_export`` – ``--command export-analyzer``: the finished dataset in
                      the LLM Log Analyzer evaluation payload format.

The review commands live in ``src.utils.helper_review`` because export/apply
share their worksheet primitives with nothing else; everything that is a
pipeline stage in its own right lives here.
"""
