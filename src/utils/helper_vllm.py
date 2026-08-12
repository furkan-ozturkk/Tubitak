"""Client for the local vLLM servers (Section 5.5).

Used by the medium tier (gold draft) and the hard tier (gold draft + per-claim
groundedness check). Concurrency is capped by a semaphore at
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

    Args:
        model_id: The model id a vLLM server was started with.

    Returns:
        A lowercase family token, e.g. ``"llama"`` or ``"qwen"``.
    """
    lowered = model_id.lower()
    for family in _KNOWN_FAMILIES:
        if family in lowered:
            return family
    return lowered


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

    def groundedness_check(
        self, claim: str, evidence_text: str
    ) -> tuple[str, dict[str, Any]]:
        """Checks whether the evidence supports one claim.

        Always uses ``groundedness_model``. Anything that is not a clear yes or
        no is recorded as ``"partial"`` rather than forced to one side: the
        verdict feeds a human review worksheet, and an ambiguous answer is a
        real signal there.

        The reviewer's own provenance block is returned rather than discarded,
        so the groundedness report can record which weights and which prompt
        produced the verdict. A report that names only the drafting model
        documents half of the integrity claim.

        Args:
            claim: One sentence from the drafted gold answer.
            evidence_text: The evidence block the claim must be supported by.

        Returns:
            Tuple ``(verdict, result)`` where verdict is ``"yes"``, ``"no"`` or
            ``"partial"``, and result carries ``text`` and ``model``.
        """
        prompt = (
            "You are a strict fact-checker. You will be given EVIDENCE (raw log lines) "
            "and one CLAIM sentence taken from a longer answer about that evidence. "
            "Decide whether the evidence supports the claim.\n\n"
            "The claim does not have to be a direct quote. A comparative or summarizing "
            "judgment (e.g. 'X is more anomalous than Y') is supported if the underlying "
            "pattern -- timing, repetition, count, duration -- is genuinely present in the "
            "evidence, even though those exact words are not. What is never supported: a "
            "specific cause, actor identity, actor category, or motive that is not itself "
            "named in the evidence (e.g. 'a botnet', 'a script kiddie', 'a misconfigured "
            "switch'), and any fact stated about one id (a block, container, node or "
            "address) that the evidence actually shows about a different id.\n\n"
            "Answer with exactly one word: yes, no, or partial.\n\n"
            f"EVIDENCE:\n{evidence_text}\n\nCLAIM:\n{claim}\n\nAnswer:"
        )
        result = self._generate(
            self.config.groundedness_base_url, self.config.groundedness_model, prompt
        )
        answer = result["text"].strip().lower()
        if answer.startswith("yes"):
            verdict = "yes"
        elif answer.startswith("no"):
            verdict = "no"
        else:
            verdict = "partial"
        return verdict, result

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
