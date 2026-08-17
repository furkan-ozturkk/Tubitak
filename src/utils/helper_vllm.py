"""Client for the local vLLM servers (Section 5.5).

Used by the medium and hard tiers (gold draft, via ``draft``) and by all three
tiers (easy, medium, hard) for the post-hoc quality check (``check_dimensions``),
run on ``groundedness_model``. Concurrency is capped by a semaphore at
``VllmConfig.max_parallel_calls``, and every call retries with exponential
backoff: a vLLM server can still be mid-restart or mid-graph-capture when a
pass starts, and a transient failure there is not a reason for a generation
pass to lose its work.

Per the Scientific Integrity Rule (Section 5.5/6) the drafting model and the
reviewing model must be from different *families*. vLLM, unlike Ollama, does
not report a model's family or content digest through its API — each server
hosts exactly one model and exposes only its served id via ``/v1/models``. The
family is therefore inferred from that id (see ``_infer_family``), and the
digest is a fingerprint of the id and the server address rather than a hash of
the weights themselves; both are documented approximations of what the
Ollama-backed version of this module could read directly from server
metadata, kept because the schema (``config/question_schema.json``) requires
a non-empty ``family`` and ``digest`` on every model-assisted record.

The separation is expressed three ways on purpose: two distinct methods
reading two distinct config fields, a name check in ``config.args``, and the
family check here.
"""

import hashlib
import json
import re
import sys
import threading
import time
import urllib.request
from typing import Any

from src.params.vllm_params import VllmConfig

_KNOWN_FAMILIES = (
    "llama",
    "qwen",
    "nemotron",
    "gpt-oss",
    "gptoss",
    "mistral",
    "gemma",
    "phi",
    "deepseek",
)

_FAMILY_VERSION_PATTERN = re.compile(
    r"(" + "|".join(_KNOWN_FAMILIES) + r")(\d+(?:\.\d+)*)?"
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _infer_family(model_id: str) -> str:
    """Derives a best-effort family label from a served model id.

    vLLM does not expose model-family metadata the way Ollama's ``/api/tags``
    does, so the family is inferred by searching the lowercased id for one of
    a handful of known family names. This is weaker than a server-reported
    field — an unlisted family falls back to the whole lowercased id, which
    guarantees two different ids are treated as different families even when
    neither matches a known name — but it is what is knowable from a served
    model id alone.

    A bare stem is not enough once two roles can both be "qwen": adding
    ``sql_invention`` (a Qwen2.5-Coder checkpoint) alongside ``groundedness``
    (a Qwen3 checkpoint) made ``"qwen" == "qwen"`` collapse two genuinely
    different models -- different generation, different training, different
    weights -- into a false "same family" that ``assert_model_families_differ``
    would then wrongly refuse. The version digits immediately following the
    stem (``qwen3``, ``qwen2.5``) are captured when present, so the two stay
    distinguishable; an id search picks whichever occurrence of the stem in
    the id carries a version (HF ids commonly repeat the family name once
    bare in the org prefix and once versioned in the repo name, e.g.
    ``"Qwen/Qwen3-32B-AWQ"``, and the bare occurrence would otherwise win by
    appearing first).

    Args:
        model_id: The model id a vLLM server was started with.

    Returns:
        A lowercase family token, e.g. ``"llama3.3"`` or ``"qwen3"`` when a
        version could be read off the id, otherwise the bare stem (or, for an
        unlisted family, the whole lowercased id).
    """
    lowered = model_id.lower()
    matches = list(_FAMILY_VERSION_PATTERN.finditer(lowered))
    if not matches:
        return lowered
    versioned = [match for match in matches if match.group(2)]
    match = versioned[0] if versioned else matches[0]
    stem, version = match.group(1), match.group(2)
    return f"{stem}{version}" if version else stem


def _strip_thinking(text: str) -> str:
    """Removes any ``<think>...</think>`` block a reasoning model emitted.

    vLLM has no per-request equivalent of Ollama's ``think: false`` option;
    ``_generate`` already asks the chat template to skip reasoning via
    ``chat_template_kwargs``, but a model that ignores the hint still wraps its
    reasoning in these tags in ``content`` rather than a separate field. Every
    caller here wants one clean answer (a draft, a yes/no/partial verdict, a
    SQL statement), so any such block is dropped before the text is used.

    Args:
        text: Raw completion text.

    Returns:
        ``text`` with every ``<think>...</think>`` span removed and the result
        stripped of leading/trailing whitespace.
    """
    while _THINK_OPEN in text and _THINK_CLOSE in text:
        start = text.find(_THINK_OPEN)
        end = text.find(_THINK_CLOSE, start)
        if end == -1:
            break
        text = text[:start] + text[end + len(_THINK_CLOSE) :]
    return text.strip()


class VllmClient:
    """Retrying, concurrency-capped client for the two role vLLM servers.

    Attributes:
        config: The connection, model roles and retry policy this client uses.
    """

    def __init__(self, config: VllmConfig):
        """Initialises the client and rejects an identical model pair by name.

        Args:
            config: Server addresses, the two role models, and retry policy.

        Raises:
            ValueError: If the gold-draft and groundedness model ids are equal.
                The family comparison needs each server's own metadata and
                therefore happens in ``assert_model_families_differ``.
        """
        if config.gold_draft_model == config.groundedness_model:
            raise ValueError(
                f"gold_draft_model and groundedness_model are both "
                f"'{config.gold_draft_model}'; Section 5.5/6 requires the drafting "
                f"model and the reviewing model to be different families."
            )
        self.config = config
        self._semaphore = threading.Semaphore(config.max_parallel_calls)
        self._served_id_cache: dict[str, str] = {}
        self._families_checked = False
        self._sql_invention_family_checked = False

    def _request(
        self, base_url: str, path: str, payload: dict | None = None, method: str = "GET"
    ) -> dict:
        """Issues one HTTP request to a server, retrying with exponential backoff.

        Args:
            base_url: The vLLM server to call.
            path: Path below the base URL, e.g. ``"/v1/models"``.
            payload: JSON body, or ``None`` for a GET.
            method: HTTP method.

        Returns:
            The decoded JSON response.

        Raises:
            RuntimeError: If every attempt failed. The last underlying error is
                included, since a DNS failure and a 500 call for different fixes.
        """
        url = base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_err = None
        with self._semaphore:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    request = urllib.request.Request(
                        url,
                        data=data,
                        method=method,
                        headers={
                            "User-Agent": "logrouter-datasetgen/helper_vllm",
                            "Content-Type": "application/json",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=1800) as response:
                        return json.loads(response.read().decode("utf-8"))
                except Exception as error:
                    last_err = error
                    if attempt < self.config.max_retries:
                        wait = self.config.backoff_base_seconds * (2 ** (attempt - 1))
                        print(
                            f"  [retry {attempt}/{self.config.max_retries}] {method} {url} "
                            f"-> {error} ; waiting {wait:.1f}s",
                            file=sys.stderr,
                        )
                        time.sleep(wait)
        raise RuntimeError(
            f"vLLM call failed ({self.config.max_retries} attempts): {method} {url} "
            f":: {last_err}"
        )

    def served_model_id(self, base_url: str) -> str:
        """Returns the model id the server at ``base_url`` was started with.

        Caches per server for the lifetime of the client: a vLLM server hosts
        exactly one model for its whole run, so the id cannot change under it.

        Args:
            base_url: The vLLM server to query.

        Returns:
            The served model id, e.g. ``"Qwen/Qwen3-32B-AWQ"``.

        Raises:
            RuntimeError: If the server reports no model at all.
        """
        if base_url in self._served_id_cache:
            return self._served_id_cache[base_url]
        payload = self._request(base_url, "/v1/models")
        models = payload.get("data", [])
        if not models:
            raise RuntimeError(f"vLLM server at {base_url} reports no served model")
        served_id = models[0]["id"]
        self._served_id_cache[base_url] = served_id
        return served_id

    def model_details(self, base_url: str, model_name: str) -> dict[str, str]:
        """Returns the inferred family and fingerprint of one role model.

        Args:
            base_url: The server that role is configured to call.
            model_name: The model id that role expects to find there.

        Returns:
            Mapping with ``name``, ``digest`` and ``family``.

        Raises:
            RuntimeError: If the server does not actually serve that model.
        """
        served_id = self.served_model_id(base_url)
        if served_id != model_name:
            raise RuntimeError(
                f"vLLM server at {base_url} serves '{served_id}', not the "
                f"configured '{model_name}'"
            )
        digest = "sha256:" + hashlib.sha256(
            f"{base_url}|{model_name}".encode("utf-8")
        ).hexdigest()
        return {"name": model_name, "digest": digest, "family": _infer_family(model_name)}

    def assert_model_families_differ(self) -> None:
        """Verifies the two role models are different families, per their ids.

        Runs once, on the first generation call. Distinct base URLs are not by
        themselves sufficient — two servers could still be started with the
        same weights — so the check compares the inferred family (and, as a
        stronger signal available here, the base URL itself) rather than
        trusting the addresses alone.

        Raises:
            RuntimeError: If the two models resolve to the same server and id,
                or share an inferred family.
        """
        if self._families_checked:
            return
        draft = self.model_details(
            self.config.gold_draft_base_url, self.config.gold_draft_model
        )
        review = self.model_details(
            self.config.groundedness_base_url, self.config.groundedness_model
        )
        if draft["digest"] == review["digest"]:
            raise RuntimeError(
                f"gold_draft_model '{draft['name']}' and groundedness_model "
                f"'{review['name']}' resolve to the same server and model; "
                f"Section 5.5/6 forbids a model certifying its own answer."
            )
        if draft["family"] == review["family"]:
            raise RuntimeError(
                f"gold_draft_model '{draft['name']}' and groundedness_model "
                f"'{review['name']}' are both family '{draft['family']}'; "
                f"Section 5.5/6 requires different model families."
            )
        self._families_checked = True

    def assert_sql_invention_differs_from_groundedness(self) -> None:
        """Verifies ``sql_invention_model`` and ``groundedness_model`` differ.

        The pairing ``assert_model_families_differ`` checks is gold-draft
        versus groundedness; SQL invention (Section 7.1,
        ``invent_sql_question``) is a third role that can be a different model
        entirely (a code-specialised checkpoint), so its own answer -- the
        invented question and query -- needs its own guarantee that the model
        checking it afterwards is not itself, by the same Section 5.5/6 logic.
        Runs once, on the first ``invent_sql_question`` call.

        Raises:
            RuntimeError: If the two models resolve to the same server and id,
                or share an inferred family.
        """
        if self._sql_invention_family_checked:
            return
        invention = self.model_details(
            self.config.sql_invention_base_url, self.config.sql_invention_model
        )
        review = self.model_details(
            self.config.groundedness_base_url, self.config.groundedness_model
        )
        if invention["digest"] == review["digest"]:
            raise RuntimeError(
                f"sql_invention_model '{invention['name']}' and "
                f"groundedness_model '{review['name']}' resolve to the same "
                f"server and model; Section 5.5/6 forbids a model certifying "
                f"its own answer."
            )
        if invention["family"] == review["family"]:
            raise RuntimeError(
                f"sql_invention_model '{invention['name']}' and "
                f"groundedness_model '{review['name']}' are both family "
                f"'{invention['family']}'; Section 5.5/6 requires different "
                f"model families."
            )
        self._sql_invention_family_checked = True

    def _generate(self, base_url: str, model: str, prompt: str) -> dict[str, Any]:
        """Runs one completion and returns it with its provenance block.

        Args:
            base_url: The vLLM server to call.
            model: Model id to run; must match what that server was started
                with.
            prompt: Full prompt text, sent as a single user turn so the
                server's own chat template wraps it the way an instruct model
                expects — the closest vLLM equivalent of Ollama's ``/api/generate``,
                which applies the model's template by default too.

        Returns:
            Mapping with ``text`` (the stripped, thinking-free completion) and
            ``model`` (the provenance block, including the prompt hash).
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.num_predict,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        result = self._request(base_url, "/v1/chat/completions", payload, method="POST")
        raw_text = result["choices"][0]["message"].get("content") or ""
        details = self.model_details(base_url, model)
        return {
            "text": _strip_thinking(raw_text),
            "model": {
                "name": model,
                "family": details["family"],
                "digest": details["digest"],
                "prompt_sha256": "sha256:"
                + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            },
        }

    def draft(self, prompt: str) -> dict[str, Any]:
        """Drafts a gold answer, always with ``gold_draft_model`` (Section 5.5).

        Args:
            prompt: Full prompt text, including the evidence block.

        Returns:
            The completion and its provenance block.

        Raises:
            RuntimeError: If the two role models turn out to share a family.
        """
        self.assert_model_families_differ()
        return self._generate(
            self.config.gold_draft_base_url, self.config.gold_draft_model, prompt
        )

    def check_dimensions(
        self,
        context: dict[str, str],
        dimensions: list[tuple[str, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Checks a produced answer against several independent quality dimensions.

        One call, always on ``groundedness_model`` -- a different family from the
        drafting model (Section 5.5/6) -- evaluates every dimension at once against
        the full context. This replaces the earlier per-sentence claim check
        (``groundedness_check``), which showed the reviewer one claim and the
        evidence but never the question the answer had to address: a claim could
        be locally true and still leave the check blind to an answer that did not
        actually answer what was asked. Every dimension defined here sees whatever
        context blocks the caller passes -- typically the question, the evidence
        and the answer together.

        Args:
            context: Named context blocks to show the model, e.g. ``{"QUESTION":
                ..., "EVIDENCE": ..., "ANSWER": ...}``, rendered in insertion
                order.
            dimensions: ``(key, question)`` pairs, one per quality dimension --
                ``key`` is the machine-readable label stored on the record,
                ``question`` is the natural-language question shown to the model.

        Returns:
            Tuple ``(checks, model)`` where ``checks`` holds one
            ``{"dimension", "verdict"[, "detail"]}`` entry per input dimension, in
            the same order, ``verdict`` being ``"yes"``, ``"no"`` or
            ``"partial"``; ``model`` is the reviewing model's provenance block.
        """
        context_text = "\n\n".join(
            f"{label}:\n{text}" for label, text in context.items()
        )
        numbered_questions = "\n".join(
            f"{index}) [{key}] {question}"
            for index, (key, question) in enumerate(dimensions, start=1)
        )
        prompt = (
            "You are a strict, independent reviewer checking a produced answer "
            "against the context it was supposed to be derived from. Answer every "
            "numbered question below about that answer, using nothing but the "
            "context given -- no outside knowledge.\n\n"
            "Answer each question on its own line, in exactly this form: "
            "'<number>) <yes|no|partial>: <one short sentence of justification>'. "
            "Use 'partial' whenever the answer is not clearly right or clearly "
            "wrong -- do not force an ambiguous case to yes or no.\n\n"
            f"{context_text}\n\nQUESTIONS:\n{numbered_questions}\n\nANSWERS:"
        )
        result = self._generate(
            self.config.groundedness_base_url, self.config.groundedness_model, prompt
        )
        checks = _parse_dimension_verdicts(result["text"], dimensions)
        return checks, result["model"]

    def write_sql(self, question: str, dataset_key: str) -> dict[str, Any]:
        """Asks the model to write the SQL that answers one question.

        The prompt carries the table shape, the dataset key and the question,
        and nothing else. It does not carry the gold answer, the literal the
        generator searched for, or the cited line numbers: a model shown the
        answer would be asked to reproduce it rather than to derive it, and
        the check would confirm only that it can copy.

        Runs on ``sql_model``, which defaults to the groundedness model and is
        therefore a different family from the model that drafts gold answers.

        Args:
            question: The question's natural-language text.
            dataset_key: The ``lines.dataset`` value to restrict the query to.

        Returns:
            Mapping with ``sql`` (the extracted statement), ``text`` (the raw
            completion) and ``model`` (the provenance block).
        """
        prompt = (
            "You write PostgreSQL queries. The only table is:\n"
            "  lines(id BIGSERIAL, dataset TEXT, line_number INTEGER, text TEXT)\n"
            "One row per log line. `text` is the raw line. `line_number` is 1-based.\n\n"
            f"Answer this question about dataset '{dataset_key}':\n{question}\n\n"
            "Rules:\n"
            f"- Restrict every query with dataset = '{dataset_key}'.\n"
            "- Match text case-insensitively unless the question demands otherwise, using "
            "ILIKE or the ~* operator. Do not use regexp_count's third argument as a case-"
            "insensitivity flag -- in PostgreSQL that argument is an integer start position, "
            "not a flags string, and passing a letter there raises a type error.\n"
            "- A 'how many' / 'total number of ... occurrences' question asks for the count "
            "of matching LINES, not the count of substring occurrences within a line -- use "
            "COUNT(*) over a WHERE ... ILIKE/~* condition, not SUM(regexp_count(...)).\n"
            "- Return exactly one row and one column: the answer itself.\n"
            "- For a counting question return the count. For a yes/no question return "
            "a boolean. For a question asking for a line, return that line's text.\n"
            "- Output only the SQL statement. No prose, no markdown fence, no semicolon."
        )
        result = self._generate(
            self.config.sql_base_url, self.config.sql_model, prompt
        )
        return {
            "sql": _extract_sql(result["text"]),
            "text": result["text"],
            "model": result["model"],
        }

    def invent_sql_question(
        self,
        dataset_key: str,
        sample_lines: list[str],
        already_asked: list[str],
        already_sql: list[str],
        polarity: str,
        suggested_mode: str,
    ) -> dict[str, Any]:
        """Invents one easy-tier question and the SQL that answers it.

        The reverse of ``write_sql``: that method is handed a question and writes
        the query; this one is handed a sample of real lines and invents both the
        question and the query, so the easy tier's question count can scale with a
        parameter (``--easy_target_total``) instead of a fixed curated literal
        list (Section 7.1). Always runs on ``sql_invention_model`` -- a role
        distinct from medium/hard's ``gold_draft_model``, since writing correct
        SQL calls for a different strength than drafting a natural-language
        synthesis -- so the later ``check_dimensions`` pass on
        ``groundedness_model`` is never a model checking its own invention
        (Section 5.5/6).

        The invented SQL is untrusted exactly like ``write_sql``'s: the caller
        must still run it through ``helper_postgres.assert_readonly_select`` and
        ``assert_scoped_to_dataset`` before ever executing it, and the executed
        result -- not this call's own claim -- is what becomes the gold answer.

        Args:
            dataset_key: The ``lines.dataset`` value the invented query must
                restrict itself to.
            sample_lines: A real excerpt of this dataset's lines, shown as
                inspiration -- never as something to copy verbatim into the
                answer, since the answer will come from executing the query
                against the whole table, not from what is shown here.
            already_asked: Questions already invented for this dataset in this
                run, so the model does not invent the same idea twice.
            already_sql: The ``ANSWER_SQL`` already accepted for this dataset
                in this run, shown alongside ``already_asked`` because a
                reworded question repeating the same underlying condition
                (Section 7.1: observed reliably from this role's
                code-specialised model, more than from the general-purpose
                model it replaced here) still gets rejected and retried by the
                caller -- showing the SQL, not just the question text, gives
                the model the more concrete thing to actually avoid.
            polarity: ``"positive"`` asks for an ordinary presence/count
                question; ``"negative"`` asks for one about the ABSENCE of a
                pattern, mirroring the presence tier's "No" requirement
                (Section 7.1) for questions a model invents rather than a
                curated literal.
            suggested_mode: One of ``"count"``, ``"presence"``, ``"line_lookup"``
                or ``"scalar"``, cycled by the caller across a dataset's slots.
                Left entirely to the model, MODE gravitates to ``scalar``
                (aggregate questions are simply more available in a raw log
                table) and the other three all but disappear; naming a
                default steers it back without removing its escape hatch for
                a genuinely better idea.

        Returns:
            Mapping with ``mode`` (``"count"``, ``"presence"``, ``"line_lookup"``
            or ``"scalar"``), ``questions`` (three phrasings of the one invented
            question), ``answer_sql``, ``evidence_sql`` (``None`` unless
            ``mode == "scalar"``), and ``model`` (the provenance block).

        Raises:
            ValueError: If the completion does not carry every required field.
            RuntimeError: If ``sql_invention_model`` and ``groundedness_model``
                turn out to share a family.
        """
        polarity_instruction = (
            "Invent a question about the ABSENCE or NEGATION of a pattern -- e.g. "
            "how many lines do NOT contain X, how many lines lack an ERROR "
            "marker, is there any line without Y -- not a plain presence/count "
            "of a pattern."
            if polarity == "negative"
            else "Invent a question about the presence, count, or a distinctive "
            "value (e.g. the longest line, the number of distinct values of "
            "something) found in this data."
        )
        avoid = (
            (
                "Questions already asked for this dataset -- invent something "
                "different (a reworded repeat of the same underlying condition "
                "is still a repeat):\n"
                + "\n".join(f"- {q}" for q in already_asked)
                + "\n\nSQL conditions already used -- do not write a condition "
                "equivalent to any of these, even with different ILIKE text or "
                "column order:\n"
                + "\n".join(f"- {s}" for s in already_sql)
            )
            if already_asked
            else "No questions have been asked for this dataset yet."
        )
        excerpt = "\n".join(sample_lines)
        prompt = (
            "You write PostgreSQL queries. The only table is:\n"
            "  lines(id BIGSERIAL, dataset TEXT, line_number INTEGER, text TEXT)\n"
            "One row per log line. `text` is the raw line. `line_number` is 1-based.\n\n"
            f"Here is a real excerpt from dataset '{dataset_key}' (for inspiration "
            "only -- your ANSWER_SQL will run against the WHOLE table, not just "
            f"this excerpt):\n{excerpt}\n\n{avoid}\n\n"
            f"{polarity_instruction}\n\n"
            "Rules:\n"
            f"- Every query MUST restrict itself with dataset = '{dataset_key}'.\n"
            "- Match text case-insensitively unless the question demands otherwise, "
            "using ILIKE or the ~* operator. Do not use regexp_count's third "
            "argument as a case-insensitivity flag -- in PostgreSQL that argument "
            "is an integer start position, not a flags string.\n"
            "- Pick exactly one MODE:\n"
            "  count      -- ANSWER_SQL is `SELECT line_number, text FROM lines "
            "WHERE <condition>`; the answer is how many rows it returns.\n"
            "  presence   -- same shape as count; the answer is whether it "
            "returns any rows at all.\n"
            "  line_lookup -- same shape as count plus `ORDER BY ... LIMIT 1`; "
            "the answer is that one row's text.\n"
            "  scalar     -- ANSWER_SQL returns exactly one row, one column (e.g. "
            "COUNT(DISTINCT ...), MAX(length(text))); you MUST also give "
            "EVIDENCE_SQL, a `SELECT line_number, text FROM lines WHERE ...` "
            "(with a LIMIT) a human could read to sanity-check the scalar.\n"
            f"- Default to MODE={suggested_mode} unless the data in front of you "
            "genuinely suggests a different mode tells a better question -- do "
            "not default to scalar just because it is the most flexible option.\n"
            "- Output in exactly this format, one field per line, no other prose, "
            "no markdown fences, no semicolons:\n"
            "MODE: <count|presence|line_lookup|scalar>\n"
            "QUESTION_1: <question>\n"
            "QUESTION_2: <the same question, differently worded>\n"
            "QUESTION_3: <the same question, differently worded again>\n"
            "ANSWER_SQL: <the query>\n"
            "EVIDENCE_SQL: <only if MODE is scalar>"
        )
        self.assert_sql_invention_differs_from_groundedness()
        result = self._generate(
            self.config.sql_invention_base_url, self.config.sql_invention_model, prompt
        )
        parsed = _parse_invented_question(result["text"])
        parsed["model"] = result["model"]
        return parsed


_VERDICT_LINE_PATTERN = re.compile(
    r"(?im)^\s*(\d+)\)\s*(yes|no|partial)\b[:\-]?\s*(.*)$"
)


def _parse_dimension_verdicts(
    text: str, dimensions: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Parses one verdict line per dimension out of a ``check_dimensions`` completion.

    Matched by each line's leading number rather than by line order, so a model
    that answers out of sequence still lands each verdict on the dimension it was
    asked about. A dimension the model never addressed with a recognisable line
    falls back to ``"partial"`` with no detail -- the same fallback the earlier
    single-claim check used for an unparseable answer: an ambiguous result is a
    signal for the human reviewer, not a reason to fail the whole check.

    Args:
        text: The reviewing model's raw completion.
        dimensions: The ``(key, question)`` pairs the prompt asked about, in the
            order they were numbered.

    Returns:
        One ``{"dimension", "verdict"[, "detail"]}`` entry per input dimension.
    """
    by_number: dict[int, tuple[str, str]] = {}
    for match in _VERDICT_LINE_PATTERN.finditer(text):
        number = int(match.group(1))
        if number not in by_number:
            by_number[number] = (match.group(2).lower(), match.group(3).strip())

    checks = []
    for index, (key, _question) in enumerate(dimensions, start=1):
        verdict, detail = by_number.get(index, ("partial", ""))
        entry: dict[str, Any] = {"dimension": key, "verdict": verdict}
        if detail:
            entry["detail"] = detail
        checks.append(entry)
    return checks


def _extract_sql(completion: str) -> str:
    """Pulls the SQL statement out of a completion.

    Models wrap SQL in a markdown fence often enough that refusing fenced output
    would fail the check for a formatting habit rather than for a wrong query.

    Args:
        completion: The raw model output.

    Returns:
        The statement, without fences or a trailing semicolon.
    """
    text = completion.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part
            if candidate.lower().startswith("sql"):
                candidate = candidate[3:]
            candidate = candidate.strip()
            if candidate:
                text = candidate
                break
    return text.strip().rstrip(";").strip()


_INVENTED_MODES = ("count", "presence", "line_lookup", "scalar")

_FIELD_PATTERN = re.compile(
    r"^\s*(MODE|QUESTION_1|QUESTION_2|QUESTION_3|ANSWER_SQL|EVIDENCE_SQL)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_invented_question(completion: str) -> dict[str, Any]:
    """Parses ``VllmClient.invent_sql_question``'s labelled-field completion.

    Splits on the fixed field labels rather than on newlines, since ``ANSWER_SQL``
    and ``EVIDENCE_SQL`` are themselves free text that could contain line breaks
    a naive per-line parser would misread as a new field.

    Args:
        completion: The raw model output.

    Returns:
        Mapping with ``mode``, ``questions`` (three strings) and ``answer_sql``,
        plus ``evidence_sql`` (``None`` unless present).

    Raises:
        ValueError: If a required field is missing, ``MODE`` is not one of the
            four known values, or ``mode == "scalar"`` without an
            ``EVIDENCE_SQL``.
    """
    matches = list(_FIELD_PATTERN.finditer(completion))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(completion)
        fields[label] = completion[start:end].strip()

    missing = [
        label
        for label in ("MODE", "QUESTION_1", "QUESTION_2", "QUESTION_3", "ANSWER_SQL")
        if not fields.get(label)
    ]
    if missing:
        raise ValueError(
            f"invented-question completion is missing field(s): {missing}"
        )

    mode = fields["MODE"].strip().lower()
    if mode not in _INVENTED_MODES:
        raise ValueError(
            f"invented-question completion has MODE={mode!r}, expected one of "
            f"{_INVENTED_MODES}"
        )
    if mode == "scalar" and not fields.get("EVIDENCE_SQL"):
        raise ValueError("MODE=scalar requires an EVIDENCE_SQL field")

    return {
        "mode": mode,
        "questions": [
            fields["QUESTION_1"],
            fields["QUESTION_2"],
            fields["QUESTION_3"],
        ],
        "answer_sql": _extract_sql(fields["ANSWER_SQL"]),
        "evidence_sql": (
            _extract_sql(fields["EVIDENCE_SQL"]) if fields.get("EVIDENCE_SQL") else None
        ),
    }


def check_server(config: VllmConfig) -> int:
    """Verifies connectivity, the required models, and the family separation.

    A missing model and an unreachable server return different codes because
    they call for different fixes: one is starting the right vLLM server on
    the expected port, the other is a network or address problem.

    Args:
        config: Server addresses and the model roles.

    Returns:
        ``0`` everything present and the families differ, ``1`` a required
        model is missing or served from the wrong address, ``2`` a server is
        unreachable, ``3`` the two role models share a family or a digest.
    """
    client = VllmClient(config)

    servers = sorted({base_url for base_url, _model, _role in config.role_models})
    print(f"Connecting to {len(servers)} vLLM server(s): {servers}")

    served: dict[str, str] = {}
    for base_url in servers:
        try:
            served[base_url] = client.served_model_id(base_url)
        except RuntimeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2

    available = set(served.values())
    print(f"Models on server(s) ({len(available)}): {sorted(available)}")

    missing = [name for name in config.required_model_names if name not in available]
    if missing:
        print(
            f"ERROR: required model(s) missing: {missing}",
            file=sys.stderr,
        )
        return 1

    try:
        client.assert_model_families_differ()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3

    print("OK: every server is reachable and serves its expected model.")
    for base_url, name, role in config.role_models:
        details = client.model_details(base_url, name)
        print(
            f"  - {name:<48} role={role:<20} family={details['family']:<10} "
            f"server={base_url}"
        )
    return 0
