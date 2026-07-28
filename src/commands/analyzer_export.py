"""Export the dataset in the LLM Log Analyzer evaluation payload format.

A library, not a script: ``main.py`` is the only executable entry point, and
``run_analyzer_export`` is what its ``export-analyzer`` command calls.

The consumer of this dataset is the analyzer repo's ``eval/runner.py``, which
does not read a flat record list. Its payload is keyed by dataset, and each
dataset carries the questions plus the pointers the consumer's own integrity
gate (``eval/dataset/question_integrity.py``) re-verifies against the corpus::

    {
      "linux": {
        "log_file": "data/corpus/loghub-2k/Linux/Linux_2k.log",
        "corpus_sha256": "sha256:...",
        "corpus_version": "<pinned commit>",
        "questions": [ ...records... ]
      },
      ...
    }

Three deliberate translations happen on the way out, and nothing else changes:

* ``log_file`` is written relative to the analyzer repo's root, following its
  ``data/corpus/loghub-2k/<Name>/<Name>_2k.log`` layout. The path is derived
  from this repo's manifest (dataset name and filename), so the export cannot
  name a corpus file the pinned manifest does not.
* ``reviewers`` drops the generating script's identity. This repo stamps the
  script into every record so a record is never authorless; the consumer's
  contract reads ``reviewers`` as *human* sign-off and requires it only on
  human-judgement gold that claims ``verified``. Exporting the script's name
  there would assert a human review that never happened — the exact claim both
  projects' integrity rules exist to prevent. Human names appended by
  ``review-apply`` survive the filter.
* Records whose review is still open are exported as they are, ``in_review``
  included. The consumer's gate is what refuses to score them; hiding them here
  would just make the export disagree with the dataset it claims to export.

Everything the two repos must already agree on — line hashes, evidence ids,
splits — is *not* translated, because translation there would paper over a
broken contract: the shared line contract is asserted by
``tests/test_line_contract.py`` and the consumer re-hashes every cited line
itself.
"""

import json
from pathlib import Path
from typing import Any

from src.params.export_params import AnalyzerExportConfig
from src.utils.helper_records import dataset_key_from_evidence, load_questions
from src.utils.helper_run import write_json

GENERATOR_REVIEWER_PREFIX = "faz"


def _human_reviewers(record: dict[str, Any]) -> list[str]:
    """Filters a record's reviewers down to human identities.

    The generating script stamps itself into ``reviewers`` (default
    ``faz1_pilot_script``); ``review-apply`` appends the deciding human. The
    consumer's contract wants only the humans.

    Args:
        record: A question record.

    Returns:
        Reviewer identities that do not name a generating script.
    """
    return [
        reviewer
        for reviewer in record.get("reviewers", [])
        if not reviewer.startswith(GENERATOR_REVIEWER_PREFIX)
    ]


def _dataset_layout(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Maps each dataset key to its analyzer-side corpus path and version.

    Args:
        manifest: This repo's parsed ``corpus_manifest.json``.

    Returns:
        Mapping from lowercase dataset key to ``log_file`` (analyzer-root
        relative) and ``corpus_version`` (the pinned commit).
    """
    commit = manifest["source"]["pinned_commit"]
    layout: dict[str, dict[str, str]] = {}
    for entry in manifest["datasets"]:
        name = entry["name"]
        layout[name.lower()] = {
            "log_file": f"data/corpus/loghub-2k/{name}/{entry['local_filename']}",
            "corpus_version": commit,
        }
    return layout


def build_payload(
    records: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Groups a flat record list into the analyzer's dataset-keyed payload.

    Args:
        records: Loaded question records (each still carrying the loader's
            ``_source_file`` annotation, which is stripped here).
        manifest: This repo's parsed ``corpus_manifest.json``.

    Returns:
        The dataset-keyed payload.

    Raises:
        ValueError: If a record cites a dataset the manifest does not pin, or
            cites no dataset at all — either way the export cannot name the
            corpus file the consumer must verify the record against.
    """
    layout = _dataset_layout(manifest)
    payload: dict[str, Any] = {}
    for record in records:
        key = dataset_key_from_evidence(record)
        if key is None:
            raise ValueError(
                f"record {record.get('id', '?')} cites no evidence, so its corpus "
                f"file cannot be named"
            )
        if key not in layout:
            raise ValueError(
                f"record {record.get('id', '?')} cites dataset '{key}', which the "
                f"manifest does not pin"
            )
        body = payload.setdefault(
            key,
            {
                "log_file": layout[key]["log_file"],
                "corpus_sha256": record["gold_provenance"]["corpus_sha256"],
                "corpus_version": layout[key]["corpus_version"],
                "questions": [],
            },
        )
        if body["corpus_sha256"] != record["gold_provenance"]["corpus_sha256"]:
            raise ValueError(
                f"record {record.get('id', '?')} carries corpus_sha256 "
                f"{record['gold_provenance']['corpus_sha256']}, but earlier "
                f"'{key}' records carry {body['corpus_sha256']}; one dataset "
                f"cannot have been built from two corpora"
            )
        exported = {
            field: value for field, value in record.items() if field != "_source_file"
        }
        exported["reviewers"] = _human_reviewers(record)
        body["questions"].append(exported)
    return payload


def run_analyzer_export(config: AnalyzerExportConfig) -> int:
    """Exports the dataset and prints what was written.

    Args:
        config: Input dataset, manifest path and output path.

    Returns:
        ``0`` written, ``2`` the dataset held no records.
    """
    print("Command      : export-analyzer")
    print(f"Dataset      : {config.dataset}")
    print(f"Manifest     : {config.manifest}")
    print(f"Output       : {config.out}")

    records = load_questions([str(config.dataset)])
    if not records:
        print("Nothing to export: the dataset holds no records.")
        return 2

    manifest = json.loads(Path(config.manifest).read_text(encoding="utf-8"))
    payload = build_payload(records, manifest)
    write_json(Path(config.out), payload)

    total = sum(len(body["questions"]) for body in payload.values())
    print(f"Exported {total} question(s) across {len(payload)} dataset(s):")
    for key in sorted(payload):
        print(
            f"  {key:10} {len(payload[key]['questions']):4}  {payload[key]['log_file']}"
        )
    return 0
