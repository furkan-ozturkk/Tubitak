"""Configuration parameters for reading the pinned LogHub corpus.

``test_fraction`` lives here rather than beside the generators because it is a
property of the corpus partition, not of any one question tier. Its value is
applied by ``src.utils.helper_splits``, which resolves the dev/test boundary over
the whole record set at once.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.helper_splits import TEST_FRACTION


@dataclass(frozen=True)
class CorpusConfig:
    """Where the corpus is read from, and how it is partitioned.

    Attributes:
        corpus_dir: Directory holding the fetched ``*_2k.log`` files.
        manifest: ``corpus_manifest.json`` path. When set, ``validate.py`` requires
            every dataset it names to be present and checked, which turns a
            silently narrowed validation run into a failure.
        test_fraction: Fraction of evidence-group components hashed into the test
            split.
    """

    corpus_dir: Path
    manifest: Path | None = None
    test_fraction: float = TEST_FRACTION


def get_corpus_params(args: Any) -> CorpusConfig:
    """Constructs a CorpusConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace containing ``corpus_dir``, ``manifest`` and
            ``test_fraction``.

    Returns:
        CorpusConfig populated from args, with the dataclass default winning
        wherever the flag was omitted.
    """
    return CorpusConfig(
        corpus_dir=args.corpus_dir,
        manifest=args.manifest,
        test_fraction=(
            CorpusConfig.test_fraction
            if args.test_fraction is None
            else args.test_fraction
        ),
    )
