"""Easy tier (Section 7.1): count / lookup / presence.

What this tier measures: simple information lookup -- can the answer be found by a
single, literal search over the raw log, with nothing to correlate and nothing to
interpret. Every gold value here is computed by SQL against loghub's ``lines``
table and every one of them is recomputed by ``validate.py`` from the same table,
which is why ``expected_answer`` never depends on a model: nothing is asserted that
a reader cannot reproduce.

Three intents, and each is emitted under three phrasings of the same question, to
satisfy the Section 2/7.4 rule that a deterministic intent must be reachable
through at least three phrasing families. All three share one ``group_id``, so
they cannot be split across dev and test — a paraphrase of a test question
sitting in dev would be a leak. Every phrasing ships: generation no longer
narrows to a first-phrasing subset, because a dataset holding one phrasing per
intent cannot measure paraphrase robustness and fails the acceptance criteria
it was built against.

A model does enter this tier once the deterministic answer already exists: every
record is run through ``VllmClient.check_dimensions`` (``src.utils.helper_validation``)
as a post-hoc sanity net -- not a source of truth, since the answer was never in the
model's hands, but a second, independent pair of eyes on whether the question reads
cleanly and the shown evidence actually looks like it supports the answer. A record
whose check comes back with any dimension marked ``"no"`` is shipped
``review_status="in_review"`` instead of ``"verified"``, so a curation mistake (a
literal matching the wrong thing, an ambiguous phrasing) surfaces immediately
instead of waiting for a worksheet cycle that easy records otherwise never go
through.
"""

import sys
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec, LiteralSpec, LookupSpec
from src.params.generation_params import EasyTierParams, GenerationConfig
from src.utils import helper_postgres
from src.utils import helper_validation
from src.utils.helper_evidence import evidence_ref, gold_provenance, slugify
from src.utils.helper_vllm import VllmClient

CREATED_BY = "src/generators/easy_tier.py@v1"

MODEL_WRITTEN_SQL_METHOD = "model_written_deterministic_sql"

QUESTIONS_PER_INVENTION = 3

MODEL_SQL_SAMPLE_SIZE = 50

MODEL_SQL_SAMPLE_CHAR_BUDGET = 8000

MODEL_SQL_INVENTION_RETRIES = 3

MODEL_SQL_MODES = ("count", "presence", "line_lookup", "scalar")

COUNT_PHRASINGS = (
    ("how-many", "How many log lines contain '{literal}'?"),
    ("count-the", "Count the lines mentioning '{literal}'."),
    ("total-of", "What is the total number of '{literal}' occurrences?"),
)

PRESENCE_PHRASINGS = (
    ("how-many", "Does the log contain any line with '{literal}'?"),
    ("count-the", "Check whether the log has a line mentioning '{literal}'."),
    (
        "imperative",
        "Tell me if there is at least one occurrence of '{literal}' in the log.",
    ),
)

LOOKUP_PHRASINGS = {
    "first": (
        ("how-many", "Show the first line that reports '{literal}'."),
        ("count-the", "Find the first log line mentioning '{literal}'."),
        ("imperative", "Give me the first line where '{literal}' appears."),
    ),
    "last": (
        ("how-many", "Show the last line that reports '{literal}'."),
        ("count-the", "Find the last log line mentioning '{literal}'."),
        ("imperative", "Give me the last line where '{literal}' appears."),
    ),
}

_ANSWER_SHAPE_BY_KIND = {
    "count": (
        "For this COUNT question, the required answer shape is a bare integer "
        "and nothing else -- judge only whether an integer was given as the "
        "kind of answer asked for, not whether that integer is itself correct."
    ),
    "presence": (
        "For this YES/NO question, 'Yes (N matching lines)' or 'No (0 matching "
        "lines)' is this evaluation's required answer format -- the "
        "parenthetical count is a mandated part of the answer shape, not an "
        "over-answer; judge only whether a verdict plus a count was given as "
        "the kind of answer asked for, not whether the verdict or count is "
        "itself correct."
    ),
    "lookup": (
        "For this line-lookup question, the required answer shape is the "
        "exact text of one log line -- judge only whether that kind of answer "
        "was given, not whether it is the right line."
    ),
    "scalar": (
        "For this question, the required answer shape is a single numeric "
        "value (e.g. a distinct count, a maximum, an average) -- judge only "
        "whether a bare number was given as the kind of answer asked for, not "
        "whether that number is itself correct."
    ),
}


def _easy_dimensions(kind: str) -> tuple:
    """Builds the easy-tier quality dimensions for one question kind.

    ``answer_matches_question`` is the one dimension worth tailoring per kind:
    a reviewer shown one shared note covering every answer shape at once
    over-generalised it (a presence-specific "Yes/No" note got applied to a
    bare count answer too), so each kind gets only the shape note that
    actually applies to it.

    Args:
        kind: ``"count"``, ``"presence"`` or ``"lookup"``.

    Returns:
        The ``(key, question)`` pairs to pass to ``check_dimensions``.
    """
    return (
        (
            "question_well_formed",
            "Is the QUESTION clearly and unambiguously phrased, and does it "
            "correctly ask about the literal or pattern it names?",
        ),
        (
            "answer_matches_question",
            "Does the ANSWER directly answer what the QUESTION asks -- as the "
            f"right kind of answer? {_ANSWER_SHAPE_BY_KIND[kind]}",
        ),
        (
            "evidence_supports_answer",
            "Does the EVIDENCE (the query, and the lines shown beside it) "
            "genuinely support the ANSWER given?",
        ),
    )


def presence_answer(count: int) -> str:
    """Renders a presence question's gold answer from its recomputed count.

    The verdict token ("Yes"/"No") leads and the match count follows in
    parentheses. The count is there because the record's ``numeric_claims``
    declares it, and the evaluation contract shared with the consumer requires
    a declared numeric value to appear in ``expected_answer`` — a "Yes" that
    hides its count asserts a number the answer text never shows. Scorers that
    only need the verdict read the leading token; ``validate.py`` and
    ``verify_answers.py`` both compare on exactly this rendering.

    Args:
        count: The recomputed number of matching lines.

    Returns:
        The gold answer text.
    """
    verdict = "Yes" if count > 0 else "No"
    plural = "" if count == 1 else "s"
    return f"{verdict} ({count} matching line{plural})"


def count_matches(
    dataset_key: str, literal: str, case_sensitive: bool
) -> tuple[int, list[int]]:
    """Counts a literal's matching lines via Postgres.

    Args:
        dataset_key: Lowercase dataset key.
        literal: Substring to match.
        case_sensitive: Whether matching is case sensitive.

    Returns:
        Tuple ``(count, matched_indices)`` where the indices are 0-based
        positions into ``CorpusView.lines`` — the SQL returns 1-based line
        numbers, and converting here keeps evidence citation at the call sites
        plain list indexing.
    """
    count, line_numbers = helper_postgres.count_literal(
        dataset_key, literal, case_sensitive
    )
    return count, [ln - 1 for ln in line_numbers]


def count_curated_intents(
    view: CorpusView, spec: DatasetSpec, params: EasyTierParams
) -> int:
    """Counts how many curated-literal easy questions one dataset will build.

    Mirrors the pruning rules ``_build_count_or_presence``/``_build_lookup``
    apply, without building any record or calling a model — used by
    ``allocate_model_sql_slots`` (Section 7.1) to know how many of a
    ``--easy_target_total`` are already spoken for before any model-invented
    question exists, since that allocation has to happen before generation
    proper starts.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        params: Easy-tier knobs, for ``min_matches``.

    Returns:
        The number of questions (three phrasings each) the curated literals
        will produce.
    """
    intents = 0
    for literal_spec in spec.count_literals:
        count, _ = count_matches(view.key, literal_spec.literal, literal_spec.case_sensitive)
        if count >= params.min_matches:
            intents += 1
    intents += len(spec.presence_literals)
    for lookup_spec in spec.lookup_specs:
        count, _ = count_matches(view.key, lookup_spec.literal, lookup_spec.case_sensitive)
        if count >= 1:
            intents += 1
    return intents * QUESTIONS_PER_INVENTION


def allocate_model_sql_slots(
    target_total: int | None, curated_counts: dict[str, int]
) -> dict[str, int]:
    """Splits a ``--easy_target_total`` gap across datasets into invention slots.

    One slot is one ``invent_sql_question`` call, worth ``QUESTIONS_PER_INVENTION``
    records (three phrasings). Every dataset is treated as having effectively
    unlimited room for model-invented questions -- unlike a curated literal or a
    medium/hard anchor, a freely-written ``WHERE`` clause is not bounded by how
    often one fixed literal occurs, so a plain even split (remainder to the first
    datasets) is enough; there is no per-dataset ceiling to respect the way
    ``_pick_anchor_occurrences``/``_select_group_sets`` have to.

    Args:
        target_total: The ``--easy_target_total`` value, or ``None`` when the
            feature is off.
        curated_counts: Each dataset's ``count_curated_intents`` result.

    Returns:
        Mapping from dataset name to invention-slot count; every value is ``0``
        when ``target_total`` is ``None`` or already met by curated questions
        alone.
    """
    names = list(curated_counts.keys())
    if target_total is None or not names:
        return {name: 0 for name in names}
    remaining_questions = max(0, target_total - sum(curated_counts.values()))
    remaining_slots = remaining_questions // QUESTIONS_PER_INVENTION
    base, extra = divmod(remaining_slots, len(names))
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(names)}


def _cap_by_char_budget(lines: list[str], budget: int = MODEL_SQL_SAMPLE_CHAR_BUDGET) -> list[str]:
    """Trims a line list to a total character budget, keeping whole lines.

    ``MODEL_SQL_SAMPLE_SIZE`` alone is not enough: it bounds line COUNT, but a
    verbose dataset's lines can be several times longer than a compact one's
    (OpenStack's request logs average ~290 characters against BGL's ~150), and
    tokens track characters, not lines. 50 OpenStack lines measured at 7681
    prompt tokens against the invention model's 8192-token context -- with the
    500-plus tokens the rest of the prompt (rules, the growing ``already_asked``
    list) and the 512 reserved for the completion still to add, that request was
    always going to be refused. A character budget applied after the line-count
    cap keeps every dataset's excerpt within a size the model can actually
    answer from, without changing anything for the datasets short lines already
    kept well under budget.

    Args:
        lines: Candidate excerpt lines, already capped by line count.
        budget: Maximum total characters to keep.

    Returns:
        A prefix of ``lines`` whose combined length does not exceed ``budget``
        -- at least one line, even if that single line alone exceeds it.
    """
    kept: list[str] = []
    total = 0
    for line in lines:
        if kept and total + len(line) > budget:
            break
        kept.append(line)
        total += len(line)
    return kept


def _sample_excerpt(lines: list[str], slot_index: int, total_slots: int) -> list[str]:
    """Returns one rotating slice of a dataset's lines, for invention prompts.

    Chunking the file into ``total_slots`` pieces and giving each invention call
    a different one (rather than the same head of the file every time) is what
    ``already_asked`` alone cannot achieve: two calls shown identical inspiration
    tend to invent the same idea even when told not to repeat themselves.

    Args:
        lines: The dataset's full corpus.
        slot_index: 0-based index of this invention call among this dataset's
            slots.
        total_slots: How many slots this dataset was allocated.

    Returns:
        Up to ``MODEL_SQL_SAMPLE_SIZE`` consecutive lines, further trimmed to
        ``MODEL_SQL_SAMPLE_CHAR_BUDGET`` total characters (``_cap_by_char_budget``)
        so a dataset with unusually long lines cannot overflow the invention
        model's context window.
    """
    if total_slots <= 0 or not lines:
        return _cap_by_char_budget(lines[:MODEL_SQL_SAMPLE_SIZE])
    chunk = max(1, len(lines) // total_slots)
    start = min(slot_index * chunk, max(0, len(lines) - 1))
    return _cap_by_char_budget(lines[start : start + MODEL_SQL_SAMPLE_SIZE])


def query_display_sql(dataset_key: str, query: dict[str, Any]) -> str:
    """Renders a ``numeric_claims[].query`` as the SQL statement it means.

    ``raw_sql`` already stores exactly this text -- written by
    ``sql_invention_model`` (Qwen2.5-Coder-14B-Instruct-AWQ) at question-invention
    time (Section 7.1), so it is returned verbatim. ``count_literal``/
    ``count_regex`` are matched in Python at generation time, but
    ``validate.py``'s re-derivation and ``helper_postgres.count_literal``/
    ``count_regex`` run the identical statement shape at validation time -- so
    rendering it here is not an invented gloss, it is the operator's own
    meaning made visible. Showing the query, model-written or not, gives the
    groundedness check one exact, compact mechanism to judge instead of only a
    fixed literal plus a raw dump of lines.

    Args:
        dataset_key: Dataset the claim's lines belong to.
        query: One ``numeric_claims[].query`` object.

    Returns:
        A single SQL statement, or ``sql`` plus a trailing ``evidence_sql``
        comment for a scalar ``raw_sql`` claim that carries both.

    Raises:
        ValueError: If ``query["operator"]`` is not one the schema defines.
    """
    operator = query["operator"]
    if operator == "raw_sql":
        text = query["sql"]
        evidence_sql = query.get("evidence_sql")
        if evidence_sql:
            text += f"\n-- evidence_sql (sample rows for illustration):\n{evidence_sql}"
        return text
    if operator == "count_literal":
        literal = query["literal"].replace("'", "''")
        op = "LIKE" if query.get("case_sensitive") else "ILIKE"
        return (
            f"SELECT line_number, text FROM lines WHERE dataset = '{dataset_key}' "
            f"AND text {op} '%{literal}%' ORDER BY line_number"
        )
    if operator == "count_regex":
        pattern = query["pattern"].replace("'", "''")
        op = "~" if query.get("case_sensitive") else "~*"
        return (
            f"SELECT line_number, text FROM lines WHERE dataset = '{dataset_key}' "
            f"AND text {op} '{pattern}' ORDER BY line_number"
        )
    raise ValueError(f"unknown numeric_claims operator for display: {operator!r}")


def _easy_evidence_text(
    kind: str,
    count: int,
    cited_lines: list[str],
    case_sensitive: bool | None,
    sql_text: str | None,
) -> str:
    """Renders the ``EVIDENCE`` block shown to the post-hoc quality check.

    The primary evidence, when ``sql_text`` is given, is the query itself: a
    compact, exact statement of the mechanism that produced the ANSWER, the
    same statement ``validate.py`` re-executes against Postgres. A handful of
    ``cited_lines`` rides alongside it, not instead of it -- a spot check that
    the query's logic really does match real corpus text, not a substitute for
    reading the query. ``sql_text`` is ``None`` only for ``line_lookup``,
    which has no ``numeric_claims`` to render.

    A zero-match presence question has no matching line to show, only a
    placeholder ref the schema requires; stating that plainly keeps the check
    from reading an unrelated decoration line as if it were meant to support
    the "No" answer. The case-sensitivity note exists because a reviewer shown
    only the literal and the lines, with no other context, judged 'ERROR' not a
    match for a search on 'Error' -- which this dataset's matching (Section
    7.1: case-insensitive unless a spec says otherwise) disagrees with; without
    the note the check does not know which convention it is being asked to
    apply.

    Args:
        kind: ``"count"``, ``"presence"``, ``"lookup"`` or ``"scalar"``.
        count: The recomputed match count (unused for lookup/scalar).
        cited_lines: A small sample of matching lines, shown beside the query.
        case_sensitive: Whether the literal was matched case-sensitively, or
            ``None`` when there is no single literal -- a model-invented
            question's own SQL decides its matching rules, so the note says
            that instead of asserting a convention that may not hold.
        sql_text: The query that computes the ANSWER, or ``None`` for
            ``line_lookup``.

    Returns:
        The evidence text.
    """
    if case_sensitive is None:
        case_note = (
            "This question was answered by a freely-written SQL query, not a "
            "single fixed literal -- that query's own condition decides case "
            "sensitivity and what counts as a match, so judge the EVIDENCE "
            "against the QUESTION's wording, not against any single literal."
        )
    else:
        case_note = (
            "Matching is case-SENSITIVE for this question."
            if case_sensitive
            else "Matching is case-INSENSITIVE for this question, so e.g. "
            "'ERROR', 'Error' and 'error' all count as the same match."
        )
    query_block = f"QUERY (computes the ANSWER when run against Postgres):\n{sql_text}\n\n" if sql_text else ""
    if kind in ("count", "presence") and count == 0:
        return (
            f"{query_block}{case_note} No line in the dataset contains this literal "
            "(0 matches from an exact substring search). The line below is shown "
            "only because the record format requires at least one evidence "
            "reference; it is unrelated to the literal and does not itself "
            f"prove anything:\n{cited_lines[0]}"
        )
    if kind in ("count", "presence"):
        return (
            f"{query_block}{case_note} There are {count} matching lines in total, "
            "which is what the QUERY above returns -- not what is counted from the "
            f"{len(cited_lines)} example line(s) below. The lines are shown only so "
            "you can spot-check that the query's condition genuinely matches real "
            "corpus text; judge the count against the QUERY's logic, not against "
            "how many lines happen to be shown:\n" + "\n".join(cited_lines)
        )
    if kind == "scalar":
        return (
            f"{query_block}{case_note} The ANSWER is the single value the QUERY "
            "above returns when executed. The line(s) below are additional "
            "evidence that the data the query targets genuinely exists in the "
            "corpus:\n" + "\n".join(cited_lines)
        )
    return f"{case_note}\n" + "\n".join(cited_lines)


def _attach_validation(
    client: VllmClient,
    config: GenerationConfig,
    record: dict[str, Any],
    kind: str,
    count: int,
    cited_lines: list[str],
    case_sensitive: bool | None,
    sql_text: str | None,
) -> None:
    """Runs the easy-tier quality check and folds its result into one record.

    Mutates ``record`` in place: sets ``validation`` and, when any dimension
    comes back ``"no"``, downgrades ``review_status`` to ``"in_review"``. Also
    writes the per-question report ``helper_review`` reads.

    Args:
        client: vLLM client; the check always runs on ``groundedness_model``.
        config: Generation config, for ``review_dir``.
        record: The just-built record; mutated in place.
        kind: ``"count"``, ``"presence"``, ``"lookup"`` or ``"scalar"``.
        count: The recomputed match count (unused for lookup/scalar).
        cited_lines: A small sample of the evidence lines, shown beside the
            query rather than in place of it.
        case_sensitive: Whether the literal was matched case-sensitively, or
            ``None`` for a model-invented question with no single literal.
        sql_text: The query that computes the ANSWER (``query_display_sql``),
            or ``None`` for ``line_lookup``, which has no query to show.
    """
    context = {
        "QUESTION": record["question"],
        "EVIDENCE": _easy_evidence_text(kind, count, cited_lines, case_sensitive, sql_text),
        "ANSWER": record["expected_answer"],
    }
    validation = helper_validation.run_checks(client, context, _easy_dimensions(kind))
    record["validation"] = validation
    if helper_validation.has_unsupported_check(validation):
        record["review_status"] = "in_review"
    group_ids = sorted({ref["group_id"] for ref in record["evidence"]["refs"]})
    helper_validation.write_report(
        config.review_dir, record["id"], group_ids, None, validation
    )


def _normalize_sql(sql: str) -> str:
    """Reduces a SQL statement to a whitespace- and case-insensitive key.

    Used only to detect a repeated invention -- the same model asked twice can
    return the identical condition reformatted across several lines, which a
    literal string comparison would miss (Section 7.1: observed with the
    code-specialised invention model, which follows the "don't repeat this
    question" instruction less reliably than the general-purpose one it
    replaced in this role).

    Args:
        sql: A candidate ``ANSWER_SQL`` statement.

    Returns:
        A collapsed, lowercased key suitable for set membership.
    """
    return " ".join(sql.lower().split())


def _invent_and_execute(
    view: CorpusView,
    client: VllmClient,
    sample: list[str],
    already_asked: list[str],
    already_sql: set[str],
    polarity: str,
    suggested_mode: str,
) -> tuple[dict[str, Any], list[tuple], list[tuple]] | None:
    """Invents one question, validates its SQL, and executes it.

    Retries the whole invent-validate-execute cycle up to
    ``MODEL_SQL_INVENTION_RETRIES`` times before giving up on this slot: an
    unparseable completion or a rejected query is at least as likely to be a
    one-off model slip as a systematic problem, the same tolerance
    ``sql_verification.py`` applies to model-written SQL it cannot use. A
    statement matching one already accepted for this dataset (``already_sql``,
    compared via ``_normalize_sql``) is treated as a failure too and retried
    the same way, rather than shipped as a second copy of a question already
    asked.

    Args:
        view: The dataset's corpus.
        client: vLLM client.
        sample: Excerpt shown to the model as inspiration.
        already_asked: Questions already invented for this dataset this run.
        already_sql: Normalized ``ANSWER_SQL`` statements already accepted for
            this dataset this run.
        polarity: ``"positive"`` or ``"negative"``, passed through to
            ``invent_sql_question``.
        suggested_mode: The mode to steer this slot towards, passed through to
            ``invent_sql_question``.

    Returns:
        Tuple ``(invented, rows, evidence_rows)`` -- ``evidence_rows`` is empty
        unless ``invented["mode"] == "scalar"`` -- or ``None`` when every
        attempt failed.
    """
    last_error: Exception | None = None
    for attempt in range(1, MODEL_SQL_INVENTION_RETRIES + 1):
        try:
            invented = client.invent_sql_question(
                view.key,
                sample,
                already_asked,
                sorted(already_sql),
                polarity,
                suggested_mode,
            )
            helper_postgres.assert_readonly_select(invented["answer_sql"])
            helper_postgres.assert_scoped_to_dataset(invented["answer_sql"], view.key)
            if _normalize_sql(invented["answer_sql"]) in already_sql:
                raise ValueError(
                    "ANSWER_SQL duplicates a statement already accepted for "
                    "this dataset this run"
                )
            rows = helper_postgres.run_readonly_query(invented["answer_sql"])
            evidence_rows: list[tuple] = []
            if invented["mode"] == "scalar":
                if len(rows) != 1 or len(rows[0]) != 1:
                    raise ValueError(
                        f"scalar ANSWER_SQL returned {len(rows)} row(s), expected 1"
                    )
                if not isinstance(rows[0][0], (int, float)):
                    raise ValueError(
                        f"scalar ANSWER_SQL returned a non-numeric value: "
                        f"{rows[0][0]!r}"
                    )
                helper_postgres.assert_readonly_select(invented["evidence_sql"])
                helper_postgres.assert_scoped_to_dataset(
                    invented["evidence_sql"], view.key
                )
                evidence_rows = helper_postgres.run_readonly_query(
                    invented["evidence_sql"]
                )
                if not evidence_rows:
                    raise ValueError("EVIDENCE_SQL returned no rows")
            else:
                for row in rows:
                    if len(row) != 2 or not isinstance(row[0], int):
                        raise ValueError(f"ANSWER_SQL row has the wrong shape: {row!r}")
                if invented["mode"] == "line_lookup" and not rows:
                    raise ValueError("line_lookup ANSWER_SQL returned no rows")
            return invented, rows, evidence_rows
        except (ValueError, RuntimeError) as error:
            last_error = error
            print(
                f"  [retry {attempt}/{MODEL_SQL_INVENTION_RETRIES}] "
                f"{view.name} model-SQL invention: {error}",
                file=sys.stderr,
            )
    print(
        f"  [prune] {view.name}: model-SQL invention failed after "
        f"{MODEL_SQL_INVENTION_RETRIES} attempts ({last_error}), skipping slot",
        file=sys.stderr,
    )
    return None


def _model_sql_records(
    view: CorpusView,
    config: GenerationConfig,
    slot_index: int,
    invented: dict[str, Any],
    rows: list[tuple[int, Any]],
    evidence_rows: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Turns one successful invention and its query results into three records.

    Args:
        view: The dataset's corpus.
        config: Generation config.
        slot_index: This dataset's 0-based invention-slot index, used in ids.
        invented: Output of ``VllmClient.invent_sql_question``.
        rows: ``answer_sql``'s result rows.
        evidence_rows: ``evidence_sql``'s result rows (only for ``scalar``).

    Returns:
        Three records sharing one ``group_id``.
    """
    mode = invented["mode"]
    group_id = f"{view.key}:modelsql:{mode}:{slot_index}"
    query: dict[str, Any] = {"operator": "raw_sql", "sql": invented["answer_sql"]}

    if mode == "scalar":
        value = rows[0][0]
        expected_answer = str(value)
        answer_type, routing_path, task = "scalar", "sql", "Aggregation"
        cited_rows = evidence_rows[:5]
        all_match_line_numbers = [row[0] for row in evidence_rows]
        query["evidence_sql"] = invented["evidence_sql"]
    elif mode == "line_lookup":
        value = None
        line_number, text = rows[0]
        expected_answer = text
        answer_type, routing_path, task = "line_lookup", "keyword", "Lookup"
        cited_rows = [(line_number, text)]
    else:
        value = len(rows)
        expected_answer = presence_answer(value) if mode == "presence" else str(value)
        answer_type, routing_path, task = mode, "sql", "Aggregation"
        cited_rows = rows[:5] if rows else [(1, view.lines[0])]
        all_match_line_numbers = [row[0] for row in rows]

    refs = [
        evidence_ref(view.key, line_number, text, group_id)
        for line_number, text in cited_rows
    ]
    evidence: dict[str, Any] = {"refs": refs}
    if mode != "line_lookup":
        evidence["query_sql"] = query_display_sql(view.key, query)

    records = []
    for phrasing_idx, question_text in enumerate(invented["questions"]):
        record: dict[str, Any] = {
            "id": f"{view.key}_v1_modelsql_{mode}_{slot_index}_{phrasing_idx}",
            "question": question_text,
            "routing_path": routing_path,
            "answer_type": answer_type,
            "task": task,
            "difficulty": "easy",
            "phrasing_family": f"model-{phrasing_idx + 1}",
            "review_status": "verified",
            "reviewers": [config.reviewer],
            "expected_answer": expected_answer,
            "gold_provenance": gold_provenance(
                method=MODEL_WRITTEN_SQL_METHOD,
                created_by=CREATED_BY,
                created_at=config.created_at,
                corpus_sha256=view.sha256,
                model=invented["model"],
            ),
            "evidence": dict(evidence),
        }
        if answer_type != "line_lookup":
            record["numeric_claims"] = [
                {
                    "value": value,
                    "all_match_line_numbers": all_match_line_numbers,
                    "query": query,
                }
            ]
        records.append(record)
    return records


def _build_model_sql_questions(
    view: CorpusView,
    config: GenerationConfig,
    params: EasyTierParams,
    client: VllmClient,
    model_slots: int,
) -> list[dict[str, Any]]:
    """Builds up to ``model_slots`` model-invented, SQL-backed easy questions.

    Unlike the curated builders above, nothing here is drawn from
    ``dataset_specs.py``: ``VllmClient.invent_sql_question`` invents both the
    question and the SQL that answers it, and every answer is still computed by
    executing that SQL against Postgres -- the model's invention is the
    mechanism, never the source of truth (Section 7.1). This is what lets
    ``--easy_target_total`` scale the easy tier's question count past what a
    fixed literal list can produce; see ``allocate_model_sql_slots``.

    Args:
        view: The dataset's corpus.
        config: Generation config, for provenance and the post-hoc QA check.
        params: Easy-tier knobs; unused directly, carried for symmetry with the
            curated builders.
        client: vLLM client; invention runs on ``gold_draft_model``, the
            post-hoc QA check on ``groundedness_model``.
        model_slots: How many questions to invent for this dataset.

    Returns:
        Up to ``model_slots * QUESTIONS_PER_INVENTION`` records.
    """
    del params
    records: list[dict[str, Any]] = []
    already_asked: list[str] = []
    already_sql: set[str] = set()
    for slot_index in range(model_slots):
        sample = _sample_excerpt(view.lines, slot_index, model_slots)
        polarity = "negative" if slot_index % 2 == 1 else "positive"
        suggested_mode = MODEL_SQL_MODES[slot_index % len(MODEL_SQL_MODES)]
        built = _invent_and_execute(
            view, client, sample, already_asked, already_sql, polarity, suggested_mode
        )
        if built is None:
            continue
        invented, rows, evidence_rows = built
        already_asked.append(invented["questions"][0])
        already_sql.add(_normalize_sql(invented["answer_sql"]))
        new_records = _model_sql_records(
            view, config, slot_index, invented, rows, evidence_rows
        )
        mode = invented["mode"]
        kind_for_qa = "lookup" if mode == "line_lookup" else mode
        count_for_qa = 0 if mode == "scalar" else len(rows)
        source_rows = evidence_rows if mode == "scalar" else rows
        evidence_lines_for_qa = [text for _line_number, text in source_rows[:5]] or [
            view.lines[0]
        ]
        sql_text_for_qa = new_records[0]["evidence"].get("query_sql") if new_records else None
        for record in new_records:
            _attach_validation(
                client,
                config,
                record,
                kind_for_qa,
                count_for_qa,
                evidence_lines_for_qa,
                None,
                sql_text_for_qa,
            )
        records.extend(new_records)
    return records


def _build_count_or_presence(
    view: CorpusView,
    literal_spec: LiteralSpec,
    kind: str,
    config: GenerationConfig,
    params: EasyTierParams,
    client: VllmClient,
) -> list[dict[str, Any]] | None:
    """Builds the three phrasings of one count or presence question.

    A count literal with too few matches is pruned: a question whose answer is
    "1" tests nothing about aggregation and its answer is unstable under any
    corpus change. A *presence* literal with zero matches is kept, because "No"
    is exactly the answer that case is curated to produce — and since there is no
    matching line to cite, its evidence anchors on line 1, which still resolves
    to a real hash-verifiable line from the same corpus (the schema requires at
    least one ref).

    Args:
        view: The dataset's corpus.
        literal_spec: The literal to count or test for.
        kind: ``"count"`` or ``"presence"``.
        config: Generation config, for the provenance stamp.
        params: Easy-tier knobs.
        client: vLLM client used for the post-hoc quality check.

    Returns:
        Three records, or ``None`` when the literal was pruned.
    """
    literal = literal_spec.literal
    count, matched_indices = count_matches(
        view.key, literal, literal_spec.case_sensitive
    )

    if kind == "count" and count < params.min_matches:
        print(
            f"  [prune] {view.name}: count literal '{literal}' has only {count} match(es) "
            f"(<{params.min_matches}), skipping",
            file=sys.stderr,
        )
        return None

    slug = slugify(literal)
    group_id = f"{view.key}:{kind}:{slug}"

    cited_indices = matched_indices[: params.max_cited_lines]
    if kind == "presence" and count == 0:
        cited_indices = [0]
    refs = [
        evidence_ref(view.key, i + 1, view.lines[i], group_id) for i in cited_indices
    ]
    cited_lines = [view.lines[i] for i in cited_indices]
    all_match_line_numbers = [i + 1 for i in matched_indices]
    query_dict = {
        "operator": "count_literal",
        "literal": literal,
        "case_sensitive": literal_spec.case_sensitive,
    }
    sql_text = query_display_sql(view.key, query_dict)

    if kind == "presence":
        answer_text = presence_answer(count)
        answer_type = "presence"
        phrasings = PRESENCE_PHRASINGS
    else:
        answer_text = str(count)
        answer_type = "count"
        phrasings = COUNT_PHRASINGS

    records = []
    for idx, (family, template) in enumerate(phrasings):
        record = {
            "id": f"{view.key}_v1_{kind}_{slug}_{idx}",
            "question": template.format(literal=literal),
            "routing_path": "sql",
            "answer_type": answer_type,
            "task": "Aggregation",
            "difficulty": "easy",
            "phrasing_family": family,
            "review_status": "verified",
            "reviewers": [config.reviewer],
            "expected_answer": answer_text,
            "gold_provenance": gold_provenance(
                method="deterministic_aggregation",
                created_by=CREATED_BY,
                created_at=config.created_at,
                corpus_sha256=view.sha256,
            ),
            "numeric_claims": [
                {
                    "value": count,
                    "all_match_line_numbers": all_match_line_numbers,
                    "query": dict(query_dict),
                }
            ],
            "evidence": {"refs": refs, "query_sql": sql_text},
        }
        _attach_validation(
            client,
            config,
            record,
            kind,
            count,
            cited_lines,
            literal_spec.case_sensitive,
            sql_text,
        )
        records.append(record)
    return records


def _build_lookup(
    view: CorpusView,
    lookup_spec: LookupSpec,
    config: GenerationConfig,
    client: VllmClient,
) -> list[dict[str, Any]] | None:
    """Builds the three phrasings of one first/last line-lookup question.

    Args:
        view: The dataset's corpus.
        lookup_spec: The literal and whether the first or last match is wanted.
        config: Generation config, for the provenance stamp.
        client: vLLM client used for the post-hoc quality check.

    Returns:
        Three records, or ``None`` when the literal matches nothing — unlike
        presence, a lookup with no match has no answer to give.
    """
    literal = lookup_spec.literal
    count, matched_indices = count_matches(
        view.key, literal, lookup_spec.case_sensitive
    )
    if count < 1:
        print(
            f"  [prune] {view.name}: lookup literal '{literal}' has 0 matches, skipping",
            file=sys.stderr,
        )
        return None

    idx = matched_indices[0] if lookup_spec.position == "first" else matched_indices[-1]
    line_number = idx + 1
    line_text = view.lines[idx]

    slug = slugify(literal) + "_" + lookup_spec.position
    group_id = f"{view.key}:lookup:{slug}"
    refs = [evidence_ref(view.key, line_number, line_text, group_id)]

    records = []
    for i, (family, template) in enumerate(LOOKUP_PHRASINGS[lookup_spec.position]):
        record = {
            "id": f"{view.key}_v1_lookup_{slug}_{i}",
            "question": template.format(literal=literal),
            "routing_path": "keyword",
            "answer_type": "line_lookup",
            "task": "Lookup",
            "difficulty": "easy",
            "phrasing_family": family,
            "review_status": "verified",
            "reviewers": [config.reviewer],
            "expected_answer": line_text,
            "gold_provenance": gold_provenance(
                method="deterministic_aggregation",
                created_by=CREATED_BY,
                created_at=config.created_at,
                corpus_sha256=view.sha256,
            ),
            "evidence": {"refs": refs},
        }
        _attach_validation(
            client,
            config,
            record,
            "lookup",
            count,
            [line_text],
            lookup_spec.case_sensitive,
            None,
        )
        records.append(record)
    return records


def build_easy_records(
    view: CorpusView,
    spec: DatasetSpec,
    config: GenerationConfig,
    client: VllmClient,
    model_slots: int = 0,
) -> list[dict[str, Any]]:
    """Builds every easy-tier record for one dataset.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        config: Generation config; its ``easy`` field carries the tier knobs.
        client: vLLM client used for the post-hoc quality check (and, when
            ``model_slots > 0``, for inventing the model-SQL questions too).
        model_slots: How many model-invented, SQL-backed questions to add on
            top of the curated ones -- ``allocate_model_sql_slots``'s per-
            dataset share of ``--easy_target_total``. ``0`` (the default)
            reproduces the tier's original, fully curated behaviour exactly.

    Returns:
        All curated count, presence and lookup records that survived pruning,
        plus any model-invented ones.
    """
    records: list[dict[str, Any]] = []
    for literal_spec in spec.count_literals:
        built = _build_count_or_presence(
            view, literal_spec, "count", config, config.easy, client
        )
        if built:
            records.extend(built)
    for literal_spec in spec.presence_literals:
        built = _build_count_or_presence(
            view, literal_spec, "presence", config, config.easy, client
        )
        if built:
            records.extend(built)
    for lookup_spec in spec.lookup_specs:
        built = _build_lookup(view, lookup_spec, config, client)
        if built:
            records.extend(built)
    if model_slots > 0:
        records.extend(
            _build_model_sql_questions(view, config, config.easy, client, model_slots)
        )
    return records
