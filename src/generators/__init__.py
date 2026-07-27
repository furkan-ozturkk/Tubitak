"""The three question-generation tiers of Section 7.

One module per tier, each exposing a ``build_*_records(view, spec, config, ...)``
function that turns one dataset's ``CorpusView`` into records. ``generate.py``
loops the datasets and merges the results; nothing here knows about the corpus
directory, the output file, or the other tiers.

``easy_tier``   – Section 7.1: count / lookup / presence. No model. Answers are
                  computed by SQL and re-computed by ``validate.py`` from the
                  same table, so records ship ``review_status=verified``.
``medium_tier`` – Section 7.2: single-event explanation over a contiguous
                  evidence window. Model-drafted, so ``in_review``.
``hard_tier``   – Section 7.3: synthesis across >=2 regex-keyed event groups,
                  model-drafted and then claim-by-claim groundedness-checked by a
                  second model family. Also ``in_review``.

The split is by tier rather than by shape of work because the tiers differ in
exactly the thing that matters — what is allowed to assert a gold answer. Only
the easy tier certifies its own output; the other two hand theirs to a human
(``src.utils.helper_review``).
"""

from src.generators.easy_tier import build_easy_records, select_official_20
from src.generators.hard_tier import build_hard_records
from src.generators.medium_tier import build_medium_records

__all__ = [
    "build_easy_records",
    "build_hard_records",
    "build_medium_records",
    "select_official_20",
]
