"""Question validation (Sections 2 and 6).

A library, not a script: ``main.py`` is the only executable entry point, and
``run_validation`` is what its ``validate`` command calls.

Five layers, ordered by how much each can prove.

1. **Schema** — does every record match ``config/question_schema.json`` exactly?
   ``additionalProperties: false`` throughout, so this catches both missing fields
   and invented ones. A ``FormatChecker`` is attached, without which the schema's
   ``format: "date-time"`` is parsed and then ignored, and ``not-a-date`` validates.

2. **Cross-record** — id uniqueness; the rule that a hard question spans at least two
   distinct evidence groups; the rule that a deterministic intent is reachable
   through at least three phrasing families (Section 2/7.4, an error under
   ``--strict``); and consistency between ``review_status`` and how the gold was
   produced.

3. **Split** — every record's stored split is recomputed from the whole record set by
   ``src.utils.helper_splits`` and compared. That function assigns one split per
   connected component of co-cited evidence groups, so this layer is what catches an
   event that reaches both dev and test.

4. **Corpus integrity** — the loaded ``lines`` table is compared against the raw
   ``*_2k.log`` file, row count and ordered content. Generation, validation and the
   manual SQL spot-check all read that table, so if the loader dropped, duplicated or
   misnumbered a line they would agree on the same wrong corpus. Nothing downstream is
   trustworthy until this passes, which is why a mismatch is an error and not a note.

5. **Answer** — every ``numeric_claims`` value is recomputed by SQL *and compared to
   ``expected_answer``*, and every ``line_lookup`` answer is compared to the line it
   cites. This is the layer the earlier validator was missing: it recomputed claims
   and never related them to the answer, so a record with ``numeric_claims.value=490``
   and ``expected_answer="999999"`` passed with zero errors. The gold answer is the
   product; a structurally valid wrong answer silently corrupts every score computed
   from it.

Layer 4 and 5 need the corpus and the database. With ``--corpus_dir`` pointing
somewhere without ``*.log`` files they cannot run, and that is reported as an error
rather than passed over: a schema-only run that prints PASSED reads as a verified
dataset.
"""

import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:
    print(
        "ERROR: the jsonschema library is not installed (see requirements.txt).",
        file=sys.stderr,
    )
    sys.exit(3)

if importlib.util.find_spec("rfc3339_validator") is None:
    print(
        "ERROR: the rfc3339-validator library is not installed (see requirements.txt). "
        "Without it jsonschema parses the schema's format: 'date-time' and then "
        "silently ignores it, so a gold_provenance.created_at of 'not-a-date' would "
        "validate cleanly — a quieter failure than refusing to run.",
        file=sys.stderr,
    )
    sys.exit(3)

from src.data.corpus_loader import (
    lines_from_bytes,
    sha256_bytes,
    sha256_file,
    sha256_line,
)
from src.generators.easy_tier import presence_answer, query_display_sql
from src.params.results_params import ValidationReport, ValidationStats
from src.params.validation_params import ValidationConfig
from src.utils import helper_postgres
from src.utils.helper_records import (
    count_by,
    dataset_key_from_evidence,
    load_questions,
    strip_source_file,
)
from src.utils.helper_run import print_findings, print_validation_stats, write_json
from src.utils.helper_splits import expected_splits

MIN_HARD_ANSWER_SENTENCES = 4
MIN_PHRASING_FAMILIES = 3
MIN_HARD_GROUPS = 2

MODEL_ASSISTED_METHOD = "independent_model_then_human"
DETERMINISTIC_METHOD = "deterministic_aggregation"
MODEL_WRITTEN_SQL_METHOD = "model_written_deterministic_sql"
EASY_METHODS = (DETERMINISTIC_METHOD, MODEL_WRITTEN_SQL_METHOD)


def build_corpus_index(corpus_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Indexes the corpus files by dataset key, with digest and parsed lines.

    Args:
        corpus_dir: Directory of ``*_2k.log`` files, or ``None``.

    Returns:
        Mapping from dataset key to its path, digest and lines; empty when
        ``corpus_dir`` is ``None`` or holds no ``*.log`` files.
    """
    index: dict[str, dict[str, Any]] = {}
    if not corpus_dir:
        return index
    for logfile in sorted(Path(corpus_dir).glob("*.log")):
        key = logfile.stem.split("_2k")[0].lower()
        data = logfile.read_bytes()
        index[key] = {
            "path": logfile,
            "sha256": sha256_bytes(data),
            "lines": lines_from_bytes(data),
        }
    return index


def check_corpus_matches_database(
    corpus_index: dict[str, dict[str, Any]], dataset_keys: set[str]
) -> list[str]:
    """Verifies the loaded ``lines`` table equals the raw file it came from.

    Compares the row count, then the ordered line numbers and text. Only the datasets
    a record actually cites are checked, because loading a corpus the questions never
    reference is not this validation's concern.

    Args:
        corpus_index: Output of ``build_corpus_index``.
        dataset_keys: Dataset keys cited by at least one record.

    Returns:
        Human-readable mismatches; empty when every checked dataset agrees.
    """
    errors: list[str] = []
    for key in sorted(dataset_keys):
        corpus = corpus_index.get(key)
        if corpus is None:
            continue
        file_lines = corpus["lines"]
        try:
            rows = helper_postgres.fetch_all_lines(key)
        except Exception as error:
            errors.append(
                f"corpus integrity [{key}]: could not read the lines table ({error})"
            )
            continue

        if len(rows) != len(file_lines):
            errors.append(
                f"corpus integrity [{key}]: the lines table holds {len(rows)} row(s) but "
                f"{corpus['path'].name} holds {len(file_lines)} line(s); every answer "
                f"computed from this table is unreliable"
            )
            continue

        for position, (line_number, text) in enumerate(rows, start=1):
            if line_number != position:
                errors.append(
                    f"corpus integrity [{key}]: expected line_number {position} at row "
                    f"{position} but found {line_number}; the table is misnumbered"
                )
                break
            if text != file_lines[position - 1]:
                errors.append(
                    f"corpus integrity [{key}]: line {position} differs between the lines "
                    f"table and {corpus['path'].name} "
                    f"(table {sha256_line(text)[:19]}, file "
                    f"{sha256_line(file_lines[position - 1])[:19]})"
                )
                break
    return errors


def _check_evidence_refs(
    refs: list[dict[str, Any]], dataset_key: str, loc: str, errors: list[str]
) -> None:
    """Re-reads every cited line from Postgres and compares its hash.

    Args:
        refs: The record's ``evidence.refs``.
        dataset_key: Dataset the record cites.
        loc: Human-readable record location, used in error text.
        errors: Error list, appended to in place.
    """
    for ref in refs:
        line_number = ref.get("line_number")
        if line_number is None or line_number < 1:
            errors.append(f"{loc}: evidence line_number={line_number} is invalid.")
            continue
        try:
            actual_line = helper_postgres.get_line(dataset_key, line_number)
        except RuntimeError:
            errors.append(
                f"{loc}: evidence line_number={line_number} not found in loghub's Postgres "
                f"lines table for dataset '{dataset_key}'."
            )
            continue
        actual_hash = sha256_line(actual_line)
        if actual_hash != ref.get("line_hash"):
            errors.append(
                f"{loc}: evidence line_hash mismatch (line {line_number}). "
                f"expected={ref.get('line_hash')} actual={actual_hash}"
            )


def _recompute_claim(
    claim: dict[str, Any],
    dataset_key: str,
    answer_type: str | None,
    loc: str,
    errors: list[str],
) -> tuple[Any, list[int]] | None:
    """Recomputes one numeric claim's value and full match list from Postgres.

    ``raw_sql`` (Section 7.1: a model-invented easy question) is re-derived by
    literally re-executing the stored statement -- through the same
    ``assert_readonly_select``/``assert_scoped_to_dataset`` guards generation
    put it through, since stored text is still untrusted input at validate
    time -- which is a stronger guarantee than the other two operators give:
    it is not a different computation that should agree, it is the identical
    query run again. A ``scalar``-answer question has no "matching lines" of
    its own, so its ``evidence_sql`` (when present) is re-run only as a sanity
    check that it still returns something, not compared row for row.

    Args:
        claim: One ``numeric_claims`` entry.
        dataset_key: Dataset the record cites.
        answer_type: The record's ``answer_type`` — ``raw_sql`` needs this to
            know whether its statement returns matching rows (count/presence)
            or a single scalar.
        loc: Human-readable record location, used in error text.
        errors: Error list, appended to in place.

    Returns:
        Tuple ``(value, line_numbers)`` — ``line_numbers`` are 1-based and
        exactly what ``all_match_line_numbers`` must equal for every operator
        except a scalar ``raw_sql``, where it is always ``[]`` (Section 6: that
        comparison is skipped by the caller) — or ``None`` when the operator is
        unknown or the query failed.
    """
    query = claim.get("query", {})
    operator = query.get("operator")
    case_sensitive = query.get("case_sensitive", False)
    if operator == "count_literal":
        return helper_postgres.count_literal(
            dataset_key, query.get("literal", ""), case_sensitive
        )
    if operator == "count_regex":
        try:
            return helper_postgres.count_regex(
                dataset_key, query.get("pattern", ""), case_sensitive
            )
        except Exception as error:
            errors.append(f"{loc}: invalid regex '{query.get('pattern')}': {error}")
            return None
    if operator == "raw_sql":
        sql = query.get("sql", "")
        try:
            helper_postgres.assert_readonly_select(sql)
            helper_postgres.assert_scoped_to_dataset(sql, dataset_key)
            rows = helper_postgres.run_readonly_query(sql)
        except Exception as error:
            errors.append(
                f"{loc}: numeric_claims.query.sql could not be re-executed: {error}"
            )
            return None
        if answer_type == "scalar":
            if len(rows) != 1 or len(rows[0]) != 1:
                errors.append(
                    f"{loc}: numeric_claims.query.sql (scalar) returned {len(rows)} "
                    f"row(s) on re-execution, expected exactly 1."
                )
                return None
            evidence_sql = query.get("evidence_sql")
            if evidence_sql:
                try:
                    helper_postgres.assert_readonly_select(evidence_sql)
                    helper_postgres.assert_scoped_to_dataset(evidence_sql, dataset_key)
                    evidence_rows = helper_postgres.run_readonly_query(evidence_sql)
                except Exception as error:
                    errors.append(
                        f"{loc}: numeric_claims.query.evidence_sql could not be "
                        f"re-executed: {error}"
                    )
                    return None
                if not evidence_rows:
                    errors.append(
                        f"{loc}: numeric_claims.query.evidence_sql returned no rows "
                        f"on re-execution."
                    )
                    return None
            return rows[0][0], []
        return len(rows), [row[0] for row in rows]
    return None


def _check_answer(
    record: dict[str, Any], dataset_key: str, loc: str, errors: list[str]
) -> None:
    """Checks the gold answer against everything that can recompute it.

    Three answer types can be settled mechanically, and each is settled here:

    ``count`` — the answer must be the recomputed number, verbatim.
    ``presence`` — the answer must be ``Yes`` exactly when the recomputed count is
    positive, so a claim of zero matches cannot ship with an answer of ``Yes``.
    ``scalar`` — a model-invented aggregate question (Section 7.1): the answer must
    equal the recomputed scalar, verbatim; there is no matching-line list to check
    against, since the claim's own SQL does not return rows to compare.
    ``line_lookup`` — the answer must equal the text of a line the record cites,
    which is what stops a lookup from citing one line and answering with another.

    A fifth check, independent of the three above, applies whenever
    ``numeric_claims`` does: ``evidence.query_sql`` (Section 7.1) must be exactly
    the statement ``query_display_sql`` renders from the first claim's own
    ``query`` -- both describe the same statement by construction at generation
    time, so a mismatch means the two were edited out of step, not that either
    alone is wrong.

    Args:
        record: The question record.
        dataset_key: Dataset the record cites.
        loc: Human-readable record location, used in error text.
        errors: Error list, appended to in place.
    """
    answer_type = record.get("answer_type")
    expected_answer = record.get("expected_answer", "")

    if answer_type in ("count", "presence", "scalar"):
        claims = record.get("numeric_claims") or []
        if not claims:
            errors.append(
                f"{loc}: answer_type={answer_type} carries no numeric_claims, so the "
                f"answer cannot be reproduced."
            )
            return
        evidence_query_sql = record.get("evidence", {}).get("query_sql")
        try:
            expected_query_sql = query_display_sql(dataset_key, claims[0].get("query", {}))
        except (KeyError, ValueError) as error:
            errors.append(f"{loc}: evidence.query_sql could not be checked: {error}")
        else:
            if evidence_query_sql != expected_query_sql:
                errors.append(
                    f"{loc}: evidence.query_sql does not match the statement "
                    f"numeric_claims[0].query means; they must describe the same "
                    f"query."
                )
        for claim in claims:
            recomputed = _recompute_claim(claim, dataset_key, answer_type, loc, errors)
            if recomputed is None:
                continue
            recomputed_value, recomputed_line_numbers = recomputed
            claimed_value = claim.get("value")
            if recomputed_value != claimed_value:
                errors.append(
                    f"{loc}: numeric_claims value could not be reproduced from Postgres "
                    f"(claimed={claimed_value} recomputed={recomputed_value}, "
                    f"operator={claim.get('query', {}).get('operator')})."
                )
                continue
            if answer_type != "scalar":
                claimed_line_numbers = claim.get("all_match_line_numbers") or []
                if sorted(claimed_line_numbers) != sorted(recomputed_line_numbers):
                    errors.append(
                        f"{loc}: numeric_claims.all_match_line_numbers could not be "
                        f"reproduced from Postgres ({len(claimed_line_numbers)} "
                        f"claimed line(s), {len(recomputed_line_numbers)} recomputed)."
                    )
                    continue
            if answer_type == "count" and expected_answer != str(recomputed_value):
                errors.append(
                    f"{loc}: expected_answer '{expected_answer}' does not match the "
                    f"recomputed count {recomputed_value}."
                )
            elif answer_type == "presence":
                should_be = presence_answer(recomputed_value)
                if expected_answer != should_be:
                    errors.append(
                        f"{loc}: expected_answer '{expected_answer}' contradicts the "
                        f"recomputed count {recomputed_value}, which requires '{should_be}'."
                    )
            elif answer_type == "scalar" and expected_answer != str(recomputed_value):
                errors.append(
                    f"{loc}: expected_answer '{expected_answer}' does not match the "
                    f"recomputed scalar {recomputed_value}."
                )

    elif answer_type == "line_lookup":
        cited_texts = []
        for ref in record.get("evidence", {}).get("refs", []):
            line_number = ref.get("line_number")
            if not isinstance(line_number, int) or line_number < 1:
                continue
            try:
                cited_texts.append(helper_postgres.get_line(dataset_key, line_number))
            except RuntimeError:
                continue
        if not cited_texts:
            errors.append(
                f"{loc}: answer_type=line_lookup cites no resolvable line, so the answer "
                f"cannot be reproduced."
            )
        elif expected_answer not in cited_texts:
            errors.append(
                f"{loc}: answer_type=line_lookup but expected_answer does not equal any "
                f"cited line's text; the answer and its evidence disagree."
            )


def _check_provenance_consistency(
    record: dict[str, Any], loc: str, errors: list[str], warnings: list[str]
) -> None:
    """Checks that how a record was produced matches what it claims about itself.

    The schema already ties the ``model`` block to the method. What it cannot express
    is the relationship between the method and the review status: a
    ``deterministic_aggregation`` record is machine-reproducible and may ship
    ``verified``, while a model-drafted one may only reach ``verified`` through a
    human, and that human has to be named in ``reviewers``. It also cannot express
    the relationship between ``validation`` and ``review_status``: a record whose
    post-hoc quality check marked any dimension unsupported (``"no"``) has an open
    problem regardless of tier, so it cannot ship ``verified`` — this is the second,
    independent guarantee behind ``easy_tier.py``'s own auto-downgrade, and the only
    enforcement for medium/hard beyond ``helper_review``'s worksheet-time refusal.

    Args:
        record: The question record.
        loc: Human-readable record location, used in error text.
        errors: Error list, appended to in place.
        warnings: Warning list, appended to in place.
    """
    method = record.get("gold_provenance", {}).get("method")
    status = record.get("review_status")
    difficulty = record.get("difficulty")
    reviewers = record.get("reviewers") or []

    if difficulty == "easy" and method not in EASY_METHODS:
        warnings.append(
            f"{loc}: difficulty=easy but gold_provenance.method='{method}'; the easy tier "
            f"is machine-computed (either a curated literal or a model-invented, "
            f"re-executed SQL statement)."
        )
    if difficulty in ("medium", "hard") and method in EASY_METHODS:
        errors.append(
            f"{loc}: difficulty={difficulty} claims method='{method}', but no "
            f"deterministic procedure produces a {difficulty}-tier answer."
        )
    if method == MODEL_ASSISTED_METHOD and status == "verified" and len(reviewers) < 2:
        errors.append(
            f"{loc}: method='{MODEL_ASSISTED_METHOD}' and review_status='verified', but "
            f"reviewers={reviewers} names no human beyond the generating script; a "
            f"model-drafted answer is only verified once a person accepted it "
            f"(Section 7.3 step 5)."
        )

    validation = record.get("validation") or {}
    unsupported = [
        check.get("dimension")
        for check in validation.get("checks", [])
        if check.get("verdict") == "no"
    ]
    if unsupported and status == "verified":
        errors.append(
            f"{loc}: review_status='verified' but validation.checks marks "
            f"{unsupported} unsupported ('no'); a record with an unresolved "
            f"quality-check failure cannot ship verified."
        )


def validate_records(
    questions: list[dict[str, Any]],
    schema: dict[str, Any],
    corpus_index: dict[str, dict[str, Any]],
    config: ValidationConfig,
) -> tuple[list[str], list[str], ValidationStats]:
    """Runs every validation layer over the loaded records.

    Args:
        questions: Loaded records, each carrying ``_source_file``.
        schema: Parsed ``question_schema.json``.
        corpus_index: Output of ``build_corpus_index``; empty disables layers 4 and 5.
        config: Validation config, for ``strict``, ``manifest`` and ``test_fraction``.

    Returns:
        Tuple ``(errors, warnings, stats)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    validator_cls = (
        getattr(jsonschema, "Draft202012Validator", None) or jsonschema.Draft7Validator
    )
    validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())

    seen_ids: dict[str, str] = {}
    group_phrasing: dict[str, set[str]] = defaultdict(set)
    cited_datasets: set[str] = set()
    locations: list[str] = []

    for index, record in enumerate(questions):
        source = record.get("_source_file", "?")
        loc = f"[{source}#{index}] id={record.get('id', '?')}"
        locations.append(loc)

        for error in validator.iter_errors(strip_source_file(record)):
            path_str = "/".join(str(part) for part in error.path)
            errors.append(f"{loc}: SCHEMA ERROR: {error.message} (path: {path_str})")

        record_id = record.get("id")
        if record_id:
            if record_id in seen_ids:
                errors.append(
                    f"{loc}: duplicate id (first seen at: {seen_ids[record_id]})"
                )
            else:
                seen_ids[record_id] = loc

        refs = record.get("evidence", {}).get("refs", [])
        group_ids_in_record = {ref["group_id"] for ref in refs if ref.get("group_id")}
        for group_id in group_ids_in_record:
            if record.get("phrasing_family") and record.get("routing_path") in (
                "sql",
                "keyword",
            ):
                group_phrasing[group_id].add(record["phrasing_family"])

        if record.get("difficulty") == "hard":
            if len(group_ids_in_record) < MIN_HARD_GROUPS:
                errors.append(
                    f"{loc}: difficulty=hard but evidence.refs only contains "
                    f"{len(group_ids_in_record)} distinct group_id(s) "
                    f"({MIN_HARD_GROUPS} required)."
                )
            sentences = len(
                re.findall(r"[.!?](\s|$)", record.get("expected_answer", ""))
            )
            if sentences < MIN_HARD_ANSWER_SENTENCES:
                warnings.append(
                    f"{loc}: difficulty=hard but expected_answer has ~{sentences} "
                    f"sentence(s) (Section 7.3 expects "
                    f">={MIN_HARD_ANSWER_SENTENCES})."
                )

        _check_provenance_consistency(record, loc, errors, warnings)

        dataset_key = dataset_key_from_evidence(record)
        if dataset_key:
            cited_datasets.add(dataset_key)

    for index, expected_split in expected_splits(
        questions, config.test_fraction
    ).items():
        stored_split = questions[index].get("split")
        if stored_split != expected_split:
            errors.append(
                f"{locations[index]}: split='{stored_split}' but this record's evidence "
                f"belongs to an event set assigned '{expected_split}'; questions derived "
                f"from linked events must share a split, or the test set leaks."
            )

    if corpus_index:
        errors.extend(check_corpus_matches_database(corpus_index, cited_datasets))

        for index, record in enumerate(questions):
            loc = locations[index]
            dataset_key = dataset_key_from_evidence(record)
            corpus = corpus_index.get(dataset_key) if dataset_key else None
            if dataset_key and corpus is None:
                errors.append(
                    f"{loc}: no corpus file found for evidence dataset key "
                    f"'{dataset_key}'; its answers cannot be verified (check "
                    f"--corpus_dir)."
                )
                continue
            if not corpus:
                continue

            refs = record.get("evidence", {}).get("refs", [])
            _check_evidence_refs(refs, dataset_key, loc, errors)

            claimed_corpus_sha = record.get("gold_provenance", {}).get("corpus_sha256")
            if claimed_corpus_sha and claimed_corpus_sha != corpus["sha256"]:
                errors.append(
                    f"{loc}: gold_provenance.corpus_sha256 does not match the fetched "
                    f"corpus (expected={claimed_corpus_sha} actual={corpus['sha256']})."
                )

            _check_answer(record, dataset_key, loc, errors)
    else:
        errors.append(
            f"corpus checks did not run: no *.log files under "
            f"--corpus_dir '{config.corpus_dir}'. Schema and cross-record layers alone "
            f"cannot establish that any gold answer is correct, so this run is not a "
            f"pass."
        )

    if config.manifest:
        errors.extend(_check_manifest_coverage(config.manifest, cited_datasets))

    thin_groups = {
        group_id: families
        for group_id, families in group_phrasing.items()
        if len(families) < MIN_PHRASING_FAMILIES
    }
    if thin_groups:
        preview = dict(list(thin_groups.items())[:10])
        extra = " ..." if len(thin_groups) > 10 else ""
        message = (
            f"{len(thin_groups)} group_id(s) do not meet the "
            f">={MIN_PHRASING_FAMILIES} phrasing-family rule: {preview}{extra}. A "
            f"full generation pass ships every phrasing, so this indicates a "
            f"hand-narrowed or truncated file rather than a normal output."
        )
        (errors if config.strict else warnings).append(message)

    stats = ValidationStats(
        total_questions=len(questions),
        unique_ids=len(seen_ids),
        distinct_group_ids=len(group_phrasing)
        or len(
            {
                ref["group_id"]
                for record in questions
                for ref in record.get("evidence", {}).get("refs", [])
                if ref.get("group_id")
            }
        ),
        by_difficulty=count_by(questions, "difficulty"),
        by_routing_path=count_by(questions, "routing_path"),
        by_split=count_by(questions, "split"),
        by_review_status=count_by(questions, "review_status"),
        hard_question_count=sum(
            1 for record in questions if record.get("difficulty") == "hard"
        ),
        datasets_covered=sorted(cited_datasets),
    )
    return errors, warnings, stats


def _check_manifest_coverage(manifest: Path, cited_datasets: set[str]) -> list[str]:
    """Requires every dataset the manifest pins to be represented in the dataset.

    Without this, pointing ``--questions`` at a single-dataset file validates cleanly
    and says nothing about the nine datasets it omits, while the printed PASSED reads
    as a verdict on the corpus as a whole.

    Args:
        manifest: ``corpus_manifest.json`` path.
        cited_datasets: Dataset keys cited by at least one record.

    Returns:
        One error per missing dataset; empty when all are covered or the manifest
        cannot be read.
    """
    try:
        pinned = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read --manifest '{manifest}': {error}"]
    expected = {entry["name"].lower() for entry in pinned.get("datasets", [])}
    missing = sorted(expected - cited_datasets)
    if not missing:
        return []
    return [
        f"--manifest pins {len(expected)} dataset(s) but the validated questions cite "
        f"none from: {missing}"
    ]


def _input_digests(
    config: ValidationConfig, questions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Records the digests of everything this validation consumed.

    A report that names only paths cannot prove which bytes it certified, and this
    project has already had result files outlive the fixes that invalidated them.

    Args:
        config: Validation config.
        questions: Loaded records, used for their source file paths.

    Returns:
        Mapping from artefact to digest.
    """
    digests: dict[str, Any] = {}
    try:
        digests["schema"] = sha256_file(Path(config.schema))
    except OSError:
        digests["schema"] = None
    sources = sorted(
        {
            record.get("_source_file")
            for record in questions
            if record.get("_source_file")
        }
    )
    digests["questions"] = {}
    for source in sources:
        try:
            digests["questions"][source] = sha256_file(Path(source))
        except OSError:
            digests["questions"][source] = None
    return digests


def run_validation(
    config: ValidationConfig, config_snapshot: dict[str, Any] | None = None
) -> int:
    """Loads, validates, prints, and writes the JSON report.

    Args:
        config: What to validate, against what, and how strictly.
        config_snapshot: The run's configuration, recorded in the report so the report
            states which code and which flags produced the verdict.

    Returns:
        A process exit code: ``0`` passed, ``1`` failed, ``2`` nothing to validate.
        The last is a distinct code on purpose — an empty glob is not a clean dataset.
    """
    print(f"Command    : validate{' --strict' if config.strict else ''}")
    print(f"Questions  : {list(config.questions)}")
    print(f"Corpus dir : {config.corpus_dir}")

    try:
        schema = json.loads(Path(config.schema).read_text(encoding="utf-8"))
        questions = load_questions(config.questions)
        if not questions:
            print(
                "ERROR: no question records found (check the --questions pattern).",
                file=sys.stderr,
            )
            return 2

        corpus_index = build_corpus_index(config.corpus_dir)
        errors, warnings, stats = validate_records(
            questions, schema, corpus_index, config
        )

        snapshot = dict(config_snapshot or {})
        snapshot["input_digests"] = _input_digests(config, questions)
        snapshot["corpus_digests"] = {
            key: entry["sha256"] for key, entry in sorted(corpus_index.items())
        }

        report = ValidationReport(
            passed=not errors,
            stats=stats,
            errors=errors,
            warnings=warnings,
            config_snapshot=snapshot,
        )

        print_validation_stats(stats)
        print_findings(errors, warnings)

        if config.report:
            write_json(Path(config.report), report.as_dict())
            print(f"\nReport written: {config.report}")

        result_word = "PASSED" if report.passed else "FAILED"
        print(
            f"\n{result_word}: {stats.total_questions} question(s), {len(errors)} "
            f"error(s), {len(warnings)} warning(s)."
        )
        return 0 if report.passed else 1
    finally:
        helper_postgres.close_connection()
