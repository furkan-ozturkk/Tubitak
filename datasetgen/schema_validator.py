#!/usr/bin/env python3
"""
schema_validator.py

Automated checker for the acceptance criteria defined in Sections 2 and 6.
Performs three layers of validation:

  1. SCHEMA: does every question record match question_schema.json exactly?
  2. CROSS-RECORD: id uniqueness, group_id<->split consistency, the rule that
     hard questions must span >=2 distinct group_ids, and the rule that
     deterministic intents need >=3 phrasing families (--strict).
  3. EVIDENCE/GROUNDEDNESS (optional, when a corpus_dir is given): whether
     every line in evidence.refs can be re-read -- from loghub's Postgres
     `lines` table via pg_client.py, not the raw corpus file -- and its
     line_hash matches, and whether the numeric claims in numeric_claims can
     be recomputed the same way. corpus_dir is still used for the overall
     gold_provenance.corpus_sha256 check, which is a whole-file hash.

Called by main.py's `validate` subcommand.
"""
import glob
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import jsonschema
except ImportError:  # pragma: no cover
    print("ERROR: the jsonschema library is not installed (see requirements.txt).", file=sys.stderr)
    sys.exit(3)

import pg_client

DEFAULT_SCHEMA = Path(__file__).parent / "question_schema.json"


def load_questions(patterns):
    records = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            p = Path(path)
            if p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError as e:
                            raise SystemExit(f"ERROR: {p}:{lineno} invalid JSON: {e}")
                        rec["_source_file"] = str(p)
                        records.append(rec)
            else:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for rec in items:
                    rec["_source_file"] = str(p)
                    records.append(rec)
    return records


def sha256_line(line):
    return "sha256:" + hashlib.sha256(line.encode("utf-8")).hexdigest()


def sha256_bytes(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def dataset_key_from_evidence(rec):
    refs = rec.get("evidence", {}).get("refs", [])
    if not refs:
        return None
    ref_id = refs[0].get("id", "")
    return ref_id.split(":", 1)[0] if ":" in ref_id else None


def build_corpus_index(corpus_dir, manifest_path):
    """Whole-file index used only for the gold_provenance.corpus_sha256 check
    (Section 6) -- per-line evidence/numeric_claims checks go through
    pg_client.py against loghub's Postgres `lines` table instead."""
    index = {}
    if not corpus_dir:
        return index
    for logfile in sorted(corpus_dir.glob("*.log")):
        key = logfile.stem.split("_2k")[0].lower()
        data = logfile.read_bytes()
        index[key] = {"path": logfile, "sha256": sha256_bytes(data)}
    return index


def count_by(questions, field):
    c = defaultdict(int)
    for r in questions:
        c[r.get(field, "?")] += 1
    return dict(c)


def validate(questions, schema, corpus_index, strict):
    errors = []
    warnings = []
    validator_cls = getattr(jsonschema, "Draft202012Validator", None) or jsonschema.Draft7Validator
    validator = validator_cls(schema)

    seen_ids = {}
    group_split = {}
    group_phrasing = defaultdict(set)

    for i, rec in enumerate(questions):
        src = rec.get("_source_file", "?")
        rid_display = rec.get("id", "?")
        loc = f"[{src}#{i}] id={rid_display}"
        clean = {k: v for k, v in rec.items() if k != "_source_file"}

        for err in validator.iter_errors(clean):
            path_str = "/".join(str(p) for p in err.path)
            errors.append(f"{loc}: SCHEMA ERROR: {err.message} (path: {path_str})")

        rid = rec.get("id")
        if rid:
            if rid in seen_ids:
                errors.append(f"{loc}: duplicate id (first seen at: {seen_ids[rid]})")
            else:
                seen_ids[rid] = loc

        refs = rec.get("evidence", {}).get("refs", [])
        group_ids_in_rec = set()
        for r in refs:
            gid = r.get("group_id")
            if gid:
                group_ids_in_rec.add(gid)

        split = rec.get("split")
        for gid in group_ids_in_rec:
            if gid in group_split and group_split[gid] != split:
                prev_split = group_split[gid]
                errors.append(
                    f"{loc}: group_id={gid} was previously seen with split={prev_split}, "
                    f"now split={split} -- questions derived from the same event must share a split."
                )
            else:
                group_split[gid] = split
            # The phrasing-diversity rule only applies to the deterministic layer (sql/keyword)
            # (Section 2/7.4); semantic/hard questions are written with a single/few phrasings.
            if rec.get("phrasing_family") and rec.get("routing_path") in ("sql", "keyword"):
                group_phrasing[gid].add(rec["phrasing_family"])

        if rec.get("difficulty") == "hard":
            if len(group_ids_in_rec) < 2:
                errors.append(
                    f"{loc}: difficulty=hard but evidence.refs only contains {len(group_ids_in_rec)} "
                    f"distinct group_id(s) (>=2 required)."
                )
            answer_text = rec.get("expected_answer", "")
            ans_len_sentences = len(re.findall(r"[.!?](\s|$)", answer_text))
            if ans_len_sentences < 4:
                warnings.append(
                    f"{loc}: difficulty=hard but expected_answer has ~{ans_len_sentences} sentence(s) "
                    f"(Section 7.3 expects >=4)."
                )

        if corpus_index:
            dkey = dataset_key_from_evidence(rec)
            corpus = corpus_index.get(dkey) if dkey else None
            if dkey and corpus is None:
                warnings.append(
                    f"{loc}: no corpus file found for evidence dataset key '{dkey}' "
                    f"(check --corpus-dir)."
                )
            elif corpus:
                for r in refs:
                    ln = r.get("line_number")
                    if ln is None or ln < 1:
                        errors.append(f"{loc}: evidence line_number={ln} is invalid.")
                        continue
                    try:
                        actual_line = pg_client.get_line(dkey, ln)
                    except RuntimeError:
                        errors.append(
                            f"{loc}: evidence line_number={ln} not found in loghub's Postgres "
                            f"lines table for dataset '{dkey}'."
                        )
                        continue
                    actual_hash = sha256_line(actual_line)
                    if actual_hash != r.get("line_hash"):
                        errors.append(
                            f"{loc}: evidence line_hash mismatch (line {ln}). "
                            f"expected={r.get('line_hash')} actual={actual_hash}"
                        )

                claimed_corpus_sha = rec.get("gold_provenance", {}).get("corpus_sha256")
                if claimed_corpus_sha and claimed_corpus_sha != corpus["sha256"]:
                    errors.append(
                        f"{loc}: gold_provenance.corpus_sha256 does not match the fetched corpus "
                        f"(expected={claimed_corpus_sha} actual={corpus['sha256']})."
                    )

                for nc in rec.get("numeric_claims", []) or []:
                    q = nc.get("query", {})
                    op = q.get("operator")
                    case_sensitive = q.get("case_sensitive", False)
                    if op == "count_literal":
                        recomputed, _ = pg_client.count_literal(dkey, q.get("literal", ""), case_sensitive)
                    elif op == "count_regex":
                        try:
                            recomputed, _ = pg_client.count_regex(dkey, q.get("pattern", ""), case_sensitive)
                        except Exception as e:  # noqa: BLE001 - surfaces bad regex from Postgres
                            errors.append(f"{loc}: invalid regex '{q.get('pattern')}': {e}")
                            continue
                    else:
                        continue
                    claimed_value = nc.get("value")
                    if recomputed != claimed_value:
                        errors.append(
                            f"{loc}: numeric_claims value could not be reproduced from Postgres "
                            f"(claimed={claimed_value} recomputed={recomputed}, operator={op})."
                        )

    thin_groups = {gid: fams for gid, fams in group_phrasing.items() if len(fams) < 3}
    if thin_groups:
        preview = dict(list(thin_groups.items())[:10])
        extra = " ..." if len(thin_groups) > 10 else ""
        msg = f"{len(thin_groups)} group_id(s) do not meet the >=3 phrasing-family rule: {preview}{extra}"
        target_list = errors if strict else warnings
        target_list.append(msg)

    stats = {
        "total_questions": len(questions),
        "unique_ids": len(seen_ids),
        "distinct_group_ids": len(group_split),
        "by_difficulty": count_by(questions, "difficulty"),
        "by_routing_path": count_by(questions, "routing_path"),
        "by_split": count_by(questions, "split"),
        "by_review_status": count_by(questions, "review_status"),
        "hard_question_count": sum(1 for r in questions if r.get("difficulty") == "hard"),
    }
    return errors, warnings, stats


def run(questions_patterns, schema_path=DEFAULT_SCHEMA, corpus_dir=None, manifest=None,
        strict=False, report_path=None):
    """End-to-end entry point used by main.py: load, validate, print, optionally
    write a JSON report. Returns 0 (passed) or 1 (failed), matching a process exit code."""
    schema = json.load(open(schema_path, encoding="utf-8"))
    questions = load_questions(questions_patterns)
    if not questions:
        print("ERROR: no question records found (check the questions pattern).", file=sys.stderr)
        return 2

    corpus_index = build_corpus_index(corpus_dir, manifest) if corpus_dir else {}
    errors, warnings, stats = validate(questions, schema, corpus_index, strict)

    print("=== STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if warnings:
        print(f"\n=== WARNINGS ({len(warnings)}) ===")
        for w in warnings[:50]:
            print(f"  - {w}")
        if len(warnings) > 50:
            print(f"  ... (+{len(warnings) - 50} more)")

    if errors:
        print(f"\n=== ERRORS ({len(errors)}) ===", file=sys.stderr)
        for e in errors[:50]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... (+{len(errors) - 50} more)", file=sys.stderr)

    passed = len(errors) == 0

    if report_path:
        report = {
            "passed": passed,
            "stats": stats,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written: {report_path}")

    result_word = "PASSED" if passed else "FAILED"
    total_q = stats["total_questions"]
    print(f"\n{result_word}: {total_q} question(s), {len(errors)} error(s), {len(warnings)} warning(s).")
    return 0 if passed else 1
