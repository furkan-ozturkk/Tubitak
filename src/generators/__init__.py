"""The three question-generation tiers of Section 7.

One module per tier, each exposing a ``build_*_records(view, spec, config, ...)``
function that turns one dataset's ``CorpusView`` into records. ``generate.py``
loops the datasets and merges the results; nothing here knows about the corpus
directory, the output file, or the other tiers.

``easy_tier``   – simple information lookup: count / lookup / presence. No model
                  drafts the answer — it is computed by SQL and re-computed by
                  ``validate.py`` from the same table, so records ship
                  ``review_status=verified`` by default. A model still runs a
                  post-hoc quality check (``src.utils.helper_validation``) over
                  the finished record and can downgrade it to ``in_review``.
``medium_tier`` – single-event-group explanation over a symmetric evidence
                  window (context before and after one anchor occurrence).
                  Model-drafted in two stages (structured extraction, then
                  narrative) and checked holistically by a second model family
                  (``src.utils.helper_validation``), so ``in_review``.
``hard_tier``   – multi-group comparison or correlation: >=2 evidence groups
                  either contrasted (comparative) or linked by a real shared
                  identifier proven in the evidence itself (correlation),
                  drafted and checked the same way as medium. Also
                  ``in_review``.

The split is by tier rather than by shape of work because the tiers differ in
exactly the thing that matters — what is allowed to assert a gold answer. Only
the easy tier certifies its own output by default; the other two hand theirs to
a human (``src.utils.helper_review``).
"""

from src.generators.easy_tier import build_easy_records
from src.generators.hard_tier import build_hard_records
from src.generators.medium_tier import build_medium_records

__all__ = [
    "build_easy_records",
    "build_hard_records",
    "build_medium_records",
]
