"""Hard tier (Section 7.3): multi-group comparison or correlation.

What this tier measures: reasoning across >=2 distinct evidence groups -- never
answerable from one group alone, which is exactly what separates it from medium.
Two different tasks live under this tier, and the dataset spec that builds a
question says which one it is asking for:

``hard_comparative`` -- two entities of the *same kind* (two source IPs, two
compute nodes, two containers), evidence-selected independently of each other, are
contrasted: which shows a more concerning pattern, and why.

``hard_correlation`` -- two entities, possibly of *different* kinds, whose link is
not assumed but *proven*: a line in the corpus exists where both entities'
identifiers appear together (Section 4.1 of the design), so the correlation being
asked about is not a coincidence of two things picked side by side, it is a
relationship the log itself states.

The evidence groups are selected before the question is drafted, not after, for
both tasks. A hard question has to correlate at least two distinct event groups --
the schema and ``validate.py`` both enforce that -- so the groups are what the
question is built around; drafting first and hunting for supporting lines
afterwards is how a "synthesis" question ends up resting on one group. Selection is
deterministic for both tasks (Section 2): comparative groups are ranked by size,
correlation pairs by their proven-link line count, and nothing is sampled, so the
same corpus yields the same groups on every run.

Each group's evidence is a *salient* sample rather than a blind first-N/last-N
slice: the group's true first and last line are always included (so the record's
evidence still carries the group's real start and end), and the lines between them
are chosen because they show an error, a repeat, a state change or a recovery --
picked by ``_select_salient_indices``, not because they happened to come first in
the file. Citing literally every line of a large group was tried once and
measurably degraded the draft (the model lost track of which lines belonged to
which group), so the total shown per group is still capped; only *which* lines
fill that cap has changed.

The closing "root-cause" instruction no longer supplies a ready-made example
sentence. An earlier revision of this prompt offered "'these errors recur without
a corrective action in between' is grounded" as a worked example, and nearly every
drafted answer -- across unrelated datasets and unrelated failure modes -- ended up
reusing close to that exact sentence: the example became the answer instead of
illustrating one. The prompt now asks the model to state a cause only when the
evidence itself supports one, and to say plainly when it does not, without ever
being shown a stock phrase to fall back on.

The drafted answer goes through a holistic quality check
(``src.utils.helper_validation``) across four dimensions -- are both groups used
correctly, is the claimed relationship actually grounded, is there invented
causality, does the answer address the question -- run by ``groundedness_model``, a
different family from the drafting model (Section 5.5/6), with the question itself
part of what the reviewing model sees.
"""

import re
import sys
from collections import defaultdict
from typing import Any

from src.data.data_factory import CorpusView
from src.data.dataset_specs import DatasetSpec, HardComparativeSpec, HardCorrelationSpec
from src.params.generation_params import GenerationConfig
from src.utils import helper_validation
from src.utils.helper_evidence import evidence_ref, gold_provenance, slugify
from src.utils.helper_vllm import VllmClient

CREATED_BY = "src/generators/hard_tier.py@v1"

HARD_PHRASING_FAMILIES = (
    "compare",
    "how-differ",
    "contrast",
    "both-appear",
    "examine-versus",
)

HARD_DIMENSIONS = (
    (
        "groups_used_correctly",
        "Does the ANSWER use facts genuinely belonging to EACH of the two "
        "groups, without attributing something true of one group's entity to "
        "the other?",
    ),
    (
        "relationship_grounded",
        "Is the comparison or relationship the ANSWER describes between the "
        "two groups actually present in the EVIDENCE, not merely asserted?",
    ),
    (
        "no_false_causality",
        "Does the ANSWER avoid naming a specific cause, actor identity, actor "
        "category, or motive that is not itself named in the EVIDENCE?",
    ),
    (
        "answers_question",
        "Does the ANSWER fully address what the QUESTION asks, including any "
        "root-cause or relationship judgment it asks for?",
    ),
)

SALIENCE_KEYWORDS = (
    "error", "fatal", "warn", "fail", "exception", "timeout", "retry",
    "reconnect", "recover", "corrected", "terminat", "clos", "expir",
    "disconnect", "denied", "invalid", "looking", "following", "leading",
)

_NUMBER_PATTERN = re.compile(r"\d+")


def _select_group_sets(
    lines: list[str], hard_spec: HardComparativeSpec, max_sets: int
) -> list[list[tuple[str, list[int]]]]:
    """Groups lines by the spec's key regex and picks non-overlapping group sets.

    Qualifying groups are ranked largest-first (key name breaking ties) and then
    chunked into consecutive, non-overlapping sets of ``hard_spec.num_groups``
    entities each: the first set gets the two largest groups, the second set the
    next two, and so on. Chunking rather than resampling means no entity is ever
    compared against itself twice, and the ranking means the richest-evidence
    comparisons are always drafted first when ``max_sets`` caps the total below
    what the corpus could support.

    Args:
        lines: The dataset's lines.
        hard_spec: The comparative spec, carrying the key regex, the minimum
            lines per group and how many groups one question needs.
        max_sets: Upper bound on how many sets to return.

    Returns:
        Up to ``max_sets`` lists of ``(key, line_indices)`` pairs, each list of
        length ``hard_spec.num_groups``; empty when the corpus does not contain
        enough qualifying groups to build even one question.
    """
    pattern = re.compile(hard_spec.extract_key_regex)
    by_key = defaultdict(list)
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            by_key[match.group("key")].append(index)

    qualifying = {
        key: indices
        for key, indices in by_key.items()
        if len(indices) >= hard_spec.min_lines_per_group
    }
    ranked = sorted(qualifying.items(), key=lambda item: (-len(item[1]), item[0]))

    sets = []
    n = hard_spec.num_groups
    for start in range(0, len(ranked), n):
        chunk = ranked[start : start + n]
        if len(chunk) < n:
            break
        sets.append(chunk)
        if len(sets) >= max_sets:
            break
    return sets


def _select_correlation_sets(
    lines: list[str], corr_spec: HardCorrelationSpec, max_pairs: int
) -> list[tuple[str, list[int], str, list[int]]]:
    """Finds correlation pairs whose link is proven by a shared line, not assumed.

    A ``(value_a, value_b)`` pair only qualifies when at least one line in the
    file matches ``key_a_regex`` and ``key_b_regex`` at once -- that line is the
    proof the two entities are genuinely related, e.g. a Hadoop line naming both
    a container id and the task attempt it was launched for. Once a pair is
    proven, its two evidence groups are every line in the file matching each
    regex's value independently (not only the proof line), so the record can show
    the entities' fuller behaviour, not just the one sentence that links them.

    Ranked by combined evidence size (values breaking ties for determinism), and
    chunked without reusing an entity already claimed by an earlier pair -- the
    same non-overlap rule ``_select_group_sets`` applies to comparative sets.

    Args:
        lines: The dataset's lines.
        corr_spec: The correlation spec, carrying both key regexes and the
            minimum lines required per side.
        max_pairs: Upper bound on how many pairs to return.

    Returns:
        Up to ``max_pairs`` tuples of ``(value_a, indices_a, value_b, indices_b)``;
        empty when no proven, sufficiently large pair exists.
    """
    pattern_a = re.compile(corr_spec.key_a_regex)
    pattern_b = re.compile(corr_spec.key_b_regex)

    by_key_a: dict[str, list[int]] = defaultdict(list)
    by_key_b: dict[str, list[int]] = defaultdict(list)
    proven_pairs: set[tuple[str, str]] = set()
    for index, line in enumerate(lines):
        match_a = pattern_a.search(line)
        match_b = pattern_b.search(line)
        if match_a:
            by_key_a[match_a.group("key")].append(index)
        if match_b:
            by_key_b[match_b.group("key")].append(index)
        if match_a and match_b:
            proven_pairs.add((match_a.group("key"), match_b.group("key")))

    candidates = []
    for value_a, value_b in proven_pairs:
        indices_a = by_key_a.get(value_a, [])
        indices_b = by_key_b.get(value_b, [])
        if (
            len(indices_a) >= corr_spec.min_lines_per_group
            and len(indices_b) >= corr_spec.min_lines_per_group
        ):
            candidates.append((value_a, indices_a, value_b, indices_b))

    ranked = sorted(
        candidates,
        key=lambda c: (-(len(c[1]) + len(c[3])), c[0], c[2]),
    )

    used_a: set[str] = set()
    used_b: set[str] = set()
    sets = []
    for value_a, indices_a, value_b, indices_b in ranked:
        if value_a in used_a or value_b in used_b:
            continue
        sets.append((value_a, indices_a, value_b, indices_b))
        used_a.add(value_a)
        used_b.add(value_b)
        if len(sets) >= max_pairs:
            break
    return sets


def _salience_category(line: str) -> str | None:
    """Returns the first ``SALIENCE_KEYWORDS`` entry a line contains, if any.

    Args:
        line: One raw log line.

    Returns:
        The matched keyword, or ``None``.
    """
    lowered = line.lower()
    for keyword in SALIENCE_KEYWORDS:
        if keyword in lowered:
            return keyword
    return None


def _normalized_template(line: str) -> str:
    """Collapses a line's digit runs so near-identical repeats compare equal.

    Args:
        line: One raw log line.

    Returns:
        The line with every digit run replaced by ``#``.
    """
    return _NUMBER_PATTERN.sub("#", line)


def _select_salient_indices(
    view: CorpusView, indices: list[int], cap: int
) -> list[int]:
    """Picks up to ``cap`` of a group's lines, favouring the ones worth reading.

    The group's true first and last line score far above anything else so they
    are always kept (the record's evidence must still carry the group's real
    start and end -- see the module docstring). Among the rest, a line scores
    higher for matching a ``SALIENCE_KEYWORDS`` category, higher still when that
    category differs from the previous *kept* line's (a signal of a state
    change), and lower on a repeat of a message already seen (digits normalised
    away first) so a long burst of identical lines does not crowd out everything
    else. Ties break on original position, which is what makes this
    deterministic across runs.

    Args:
        view: The dataset's corpus.
        indices: Every line index in the group, ascending.
        cap: Maximum number of indices to return.

    Returns:
        Up to ``cap`` indices, ascending; all of ``indices`` unchanged when it
        already fits within ``cap``.
    """
    if len(indices) <= cap:
        return indices

    last_position = len(indices) - 1
    previous_category: str | None = None
    seen_templates: dict[str, int] = {}
    scored = []
    for position, idx in enumerate(indices):
        line = view.lines[idx]
        score = 0.0
        if position in (0, last_position):
            score += 100.0
        category = _salience_category(line)
        if category is not None:
            score += 2.0
            if category != previous_category:
                score += 1.0
            previous_category = category
        template = _normalized_template(line)
        if seen_templates.get(template, 0) >= 1:
            score -= 1.0
        seen_templates[template] = seen_templates.get(template, 0) + 1
        scored.append((score, position, idx))

    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:cap]
    return sorted(idx for _score, _position, idx in top)


def _build_evidence_block(
    view: CorpusView,
    key: str,
    indices: list[int],
    cap: int,
    group_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Builds one group's evidence refs and the text block shown to the model.

    Args:
        view: The dataset's corpus.
        key: The group key (a block id, container id, source address, ...).
        indices: Every line index in the group.
        cap: Maximum number of lines to cite for this group.
        group_id: The group's evidence identifier.

    Returns:
        Tuple ``(refs, text_block)``. A gap between two shown lines that are
        not adjacent within the group is marked in the text so the model is
        never shown a false impression of contiguity, and no count of the
        omitted lines is ever stated (Section 7.3: revealing a total teaches
        the model to compare groups by size instead of by content).
    """
    kept = _select_salient_indices(view, indices, cap)
    refs = [evidence_ref(view.key, i + 1, view.lines[i], group_id) for i in kept]

    position_by_index = {idx: position for position, idx in enumerate(indices)}
    pieces = []
    previous_position = None
    for idx in kept:
        position = position_by_index[idx]
        if previous_position is not None and position != previous_position + 1:
            pieces.append("  ... (more lines follow between these) ...")
        pieces.append(view.lines[idx])
        previous_position = position

    summary = f"[Group: {key}] {len(kept)} line(s):"
    return refs, f"{summary}\n" + "\n".join(pieces)


def _build_prompt(
    dataset_name: str,
    group_count: int,
    question_text: str,
    evidence_text: str,
    min_sentences: int,
) -> str:
    """Builds the drafting prompt for one hard question.

    Args:
        dataset_name: Dataset the groups came from.
        group_count: How many event groups the answer must use.
        question_text: The rendered question.
        evidence_text: Every group's summary and evidence block, concatenated.
        min_sentences: Minimum sentences demanded of the answer.

    Returns:
        The full prompt.
    """
    return (
        f"You are analyzing raw {dataset_name} log lines from {group_count} related event "
        "groups. Each group is a header naming the group followed by a curated sample of "
        "its lines -- the group's true first and last line are always included, and the "
        "lines between them were chosen because they show an error, a repeated failure, a "
        "state change, or a recovery, not because they happened to come first in the file. "
        "Read ONLY the evidence below. Groups may hold very different numbers of matching "
        "lines in the underlying log -- that difference is not itself a sign of anomaly, so "
        "do not compare groups by which one has more lines shown; compare them by the "
        "timing, actors and event pattern the lines actually describe. Never attribute a "
        "fact about one entity (an IP, a block id, a container id, a node) to a different "
        "entity, even one of the same kind -- two ids are two different things unless a "
        "line explicitly connects them.\n\n"
        f"Write an answer of at least {min_sentences} sentences that correlates the groups "
        "and explicitly uses facts from every group. If the evidence itself supports a "
        "specific explanation for the pattern shown -- a repeating failure, an escalation, "
        "a clear before/after state change -- end with one sentence stating it, grounded in "
        "what these particular lines show and nothing else. If the evidence does not "
        "support a specific explanation, say so plainly instead of inventing one -- do not "
        "reuse a stock closing sentence between different questions. Naming a specific "
        "unobserved cause ('a botnet', 'a misconfigured switch', 'a software bug') or a "
        "category of actor not shown in the evidence ('a script kiddie') is never grounded "
        "and must be avoided. Do not add sections, headers, bullet points, or a list of "
        "recommendations -- plain prose sentences only, nothing beyond the analysis asked "
        f"for.\n\nQUESTION: {question_text}\n\nEVIDENCE:\n{evidence_text}\n\nAnswer:"
    )


def _build_records_for_sets(
    view: CorpusView,
    spec_id: str,
    task: str,
    question_templates: tuple,
    sets: list[list[tuple[str, list[int]]]],
    config: GenerationConfig,
    client: VllmClient,
) -> list[dict[str, Any]]:
    """Drafts and checks one record per selected group-pair, comparative or correlation alike.

    Shared by both hard tasks: by the time a caller has reduced its selection
    down to ``sets`` of ``(key, line_indices)`` pairs, drafting, checking and
    record assembly no longer differ between "two same-kind entities contrasted"
    and "two linked entities correlated" -- only how the pairs were chosen did.

    Args:
        view: The dataset's corpus.
        spec_id: The spec's stable identifier, used in ids and report filenames.
        task: Task label recorded on every record (e.g. ``"Comparison"`` or
            ``"Correlation"``).
        question_templates: Phrasings cycled across this spec's sets.
        sets: Selected group-pairs, each a 2-list of ``(key, line_indices)``.
        config: Generation config; its ``hard`` field carries the tier knobs.
        client: vLLM client used for the draft and the quality check.

    Returns:
        The drafted records, all ``review_status=in_review``.
    """
    params = config.hard
    cap = params.evidence_lines_per_side * 2
    records = []
    for occurrence, selected in enumerate(sets):
        keys = [key for key, _ in selected]
        group_ids = [
            f"{view.key}:hard:{spec_id}:{occurrence}:{slugify(key)}" for key in keys
        ]
        all_refs: list[dict[str, Any]] = []
        evidence_blocks = []
        for (key, indices), group_id in zip(selected, group_ids):
            refs, block = _build_evidence_block(view, key, indices, cap, group_id)
            all_refs.extend(refs)
            evidence_blocks.append(block)
        evidence_text = "\n\n".join(evidence_blocks)

        phrasing_index = occurrence % len(question_templates)
        question_text = question_templates[phrasing_index].format(
            **{f"key{index}": key for index, key in enumerate(keys)}
        )
        phrasing_family = HARD_PHRASING_FAMILIES[
            phrasing_index % len(HARD_PHRASING_FAMILIES)
        ]

        draft = client.draft(
            _build_prompt(
                view.name, len(keys), question_text, evidence_text, params.min_sentences
            )
        )
        answer_text = draft["text"]

        question_id = f"{view.key}_v1_hard_{spec_id}_{occurrence}"
        context = {
            "QUESTION": question_text,
            "EVIDENCE": evidence_text,
            "ANSWER": answer_text,
        }
        validation = helper_validation.run_checks(client, context, HARD_DIMENSIONS)
        helper_validation.write_report(
            config.review_dir, question_id, group_ids, draft["model"], validation
        )

        records.append(
            {
                "id": question_id,
                "question": question_text,
                "routing_path": "semantic",
                "answer_type": "synthesis",
                "task": task,
                "difficulty": "hard",
                "phrasing_family": phrasing_family,
                "review_status": "in_review",
                "reviewers": [config.reviewer],
                "expected_answer": answer_text,
                "gold_provenance": gold_provenance(
                    method="independent_model_then_human",
                    created_by=CREATED_BY,
                    created_at=config.created_at,
                    corpus_sha256=view.sha256,
                    model=draft["model"],
                ),
                "evidence": {"refs": all_refs},
                "validation": validation,
            }
        )
        print(
            f"  [{view.name}] drafted hard question '{spec_id}' #{occurrence} "
            f"(task={task}, groups={keys})"
        )
    return records


def build_hard_records(
    view: CorpusView,
    spec: DatasetSpec,
    config: GenerationConfig,
    client: VllmClient,
) -> list[dict[str, Any]]:
    """Builds every hard-tier record for one dataset, comparative and correlation alike.

    A spec whose groups cannot be filled is reported as ``quota_unmet`` and skipped
    rather than relaxed: lowering ``min_lines_per_group`` to make a question fit
    would produce a correlation or comparison question over too little evidence.

    Args:
        view: The dataset's corpus.
        spec: The dataset's curation spec.
        config: Generation config; its ``hard`` field carries the tier knobs and its
            ``review_dir`` is where validation reports are written.
        client: vLLM client used for the draft and the quality check.

    Returns:
        The drafted records, all ``review_status=in_review``.
    """
    params = config.hard
    records: list[dict[str, Any]] = []

    for hard_spec in spec.hard_comparative:
        sets = _select_group_sets(view.lines, hard_spec, params.pairs_per_dataset)
        if not sets:
            print(
                f"  [quota_unmet] {view.name}/{hard_spec.spec_id}: fewer than "
                f"{hard_spec.num_groups} groups with >={hard_spec.min_lines_per_group} lines",
                file=sys.stderr,
            )
            continue
        records.extend(
            _build_records_for_sets(
                view,
                hard_spec.spec_id,
                hard_spec.task,
                hard_spec.question_templates,
                sets,
                config,
                client,
            )
        )

    for corr_spec in spec.hard_correlation:
        raw_sets = _select_correlation_sets(
            view.lines, corr_spec, params.pairs_per_dataset
        )
        if not raw_sets:
            print(
                f"  [quota_unmet] {view.name}/{corr_spec.spec_id}: no proven "
                f"correlation pair with >={corr_spec.min_lines_per_group} lines per side",
                file=sys.stderr,
            )
            continue
        sets = [
            [(value_a, indices_a), (value_b, indices_b)]
            for value_a, indices_a, value_b, indices_b in raw_sets
        ]
        records.extend(
            _build_records_for_sets(
                view,
                corr_spec.spec_id,
                corr_spec.task,
                corr_spec.question_templates,
                sets,
                config,
                client,
            )
        )

    return records
