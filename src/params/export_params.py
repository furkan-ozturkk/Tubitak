"""Configuration parameters for the ``export-analyzer`` command.

One dataclass, three paths: which dataset to export, which manifest names the
corpus files the payload must point at, and where the payload is written. The
manifest is required rather than optional here — unlike ``validate``, which can
still check schemas without one, an export without the manifest cannot name a
single ``log_file`` and would have nothing correct to write.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_EXPORT_OUT = Path("/output/pilot/questions_analyzer.json")


@dataclass(frozen=True)
class AnalyzerExportConfig:
    """``export-analyzer``: the dataset out in the analyzer's payload format.

    Attributes:
        dataset: Question dataset file to export.
        manifest: This repo's ``corpus_manifest.json``, whose dataset names and
            filenames become the payload's ``log_file`` pointers.
        out: Where the dataset-keyed payload is written.
    """

    dataset: Path
    manifest: Path
    out: Path = DEFAULT_EXPORT_OUT


def get_analyzer_export_params(args: Any) -> AnalyzerExportConfig:
    """Constructs an ExportLlaConfig from parsed command-line arguments.

    Args:
        args: Parsed argument namespace.

    Returns:
        AnalyzerExportConfig populated from args, with the dataclass default winning
        where the flag was omitted.
    """
    return AnalyzerExportConfig(
        dataset=args.dataset,
        manifest=args.manifest,
        out=(AnalyzerExportConfig.out if args.export_out is None else args.export_out),
    )
