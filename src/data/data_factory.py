"""Single provider that hands a generator one dataset's corpus.

``corpus_provider()`` is the only place a ``*_2k.log`` file is opened during
generation. Before it existed, each of the three tiers repeated the same four
steps — read the bytes, hash the file, split the lines, lowercase the name — and
a tier that drifted on any one of them would have stamped a ``corpus_sha256`` or
a line number that no longer described the file its evidence pointed into.
"""

from dataclasses import dataclass
from pathlib import Path

from src.data.corpus_loader import dataset_key, lines_from_bytes, sha256_bytes
from src.data.dataset_specs import DatasetSpec
from src.params.corpus_params import CorpusConfig


@dataclass(frozen=True)
class CorpusView:
    """One LogHub dataset, as the generators see it.

    Attributes:
        name: Dataset name as declared in ``dataset_specs`` (e.g. ``"HDFS"``).
        key: Lowercase key used in evidence ids and in Postgres ``lines.dataset``.
        path: The ``*_2k.log`` file this view was built from.
        lines: The file's lines; index ``i`` is line ``i + 1``.
        sha256: Whole-file digest, stamped into every record's
            ``gold_provenance.corpus_sha256``.
        test_fraction: The dev/test partition this run uses.
    """

    name: str
    key: str
    path: Path
    lines: list[str]
    sha256: str
    test_fraction: float

    def line(self, line_number: int) -> str:
        """Returns the text of a 1-based line number.

        Args:
            line_number: 1-based line number, as it appears in ``evidence.refs``.

        Returns:
            The line's text.
        """
        return self.lines[line_number - 1]


def corpus_provider(config: CorpusConfig, spec: DatasetSpec) -> CorpusView:
    """Loads one dataset's corpus file into a CorpusView.

    Args:
        config: Corpus configuration; supplies the directory and the dev/test
            fraction the view carries down to the generators.
        spec: The dataset's curation spec, which names the log file.

    Returns:
        A CorpusView for that dataset.

    Raises:
        FileNotFoundError: If the log file is not in ``config.corpus_dir``,
            which means the loghub volume was not mounted or the fetch did not
            complete.
    """
    log_path = config.corpus_dir / spec.log_filename
    data = log_path.read_bytes()
    return CorpusView(
        name=spec.name,
        key=dataset_key(spec.name),
        path=log_path,
        lines=lines_from_bytes(data),
        sha256=sha256_bytes(data),
        test_fraction=config.test_fraction,
    )
