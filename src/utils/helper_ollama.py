"""Client for the remote Ollama server (Section 5.5).

Used by the medium tier (gold draft) and the hard tier (gold draft + per-claim
groundedness check). Concurrency is capped by a semaphore at
``OllamaConfig.max_parallel_calls``, and every call retries with exponential
backoff: the server is user-provided and pre-existing, so a transient failure
there is not a reason for a generation pass to lose its work.

Per the Scientific Integrity Rule (Section 5.5/6) the drafting model and the
reviewing model must be from different *families*, and this module checks the
family, not just the name. Two different tags can resolve to the same weights or
the same base family, so comparing names alone would let a run pass the integrity
rule while a model effectively reviewed itself. Because families are only knowable
from the server's own metadata, the check runs on first use rather than at
construction, and it refuses the pair outright.

The separation is expressed three ways on purpose: two distinct methods reading two
distinct config fields, a name check in ``config.args``, and the family check here.
"""

import hashlib
import json
import sys
import threading
import time
import urllib.request
from typing import Any

from src.params.ollama_params import OllamaConfig


class OllamaClient:
    """Retrying, concurrency-capped client for one Ollama server.

    Attributes:
        config: The connection, model roles and retry policy this client uses.
    """

    def __init__(self, config: OllamaConfig):
        """Initialises the client and rejects an identical model pair by name.

        Args:
            config: Server address, the two role models, and retry policy.

        Raises:
            ValueError: If the gold-draft and groundedness model names are equal.
                The family comparison needs server metadata and therefore happens
                in ``assert_model_families_differ``.
        """
        if config.gold_draft_model == config.groundedness_model:
            raise ValueError(
                f"gold_draft_model and groundedness_model are both "
                f"'{config.gold_draft_model}'; Section 5.5/6 requires the drafting "
                f"model and the reviewing model to be different families."
            )
        self.config = config
        self._semaphore = threading.Semaphore(config.max_parallel_calls)
        self._model_cache: dict[str, dict] | None = None
        self._families_checked = False

    def _request(
        self, path: str, payload: dict | None = None, method: str = "GET"
    ) -> dict:
        """Issues one HTTP request to the server, retrying with exponential backoff.

        Args:
            path: Path below the base URL, e.g. ``"/api/tags"``.
            payload: JSON body, or ``None`` for a GET.
            method: HTTP method.

        Returns:
            The decoded JSON response.

        Raises:
            RuntimeError: If every attempt failed. The last underlying error is
                included, since a DNS failure and a 500 call for different fixes.
        """
        url = self.config.base_url.rstrip("/") + path
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
                            "User-Agent": "logrouter-datasetgen/helper_ollama",
                            "Content-Type": "application/json",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:
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
            f"Ollama call failed ({self.config.max_retries} attempts): {method} {url} "
            f":: {last_err}"
        )

    def list_model_names(self) -> set[str]:
        """Returns the names of every model the server reports.

        Returns:
            Model names from ``/api/tags``.
        """
        payload = self._request("/api/tags")
        self._model_cache = {
            model["name"]: model for model in payload.get("models", [])
        }
        return set(self._model_cache)

    def model_details(self, model_name: str) -> dict[str, str]:
        """Returns the digest and family of one model, caching the catalogue.

        The digest is recorded in every model-assisted record's
        ``gold_provenance.model`` block: a model name is a tag that can be
        repointed, whereas the digest identifies the weights that produced the
        answer.

        Args:
            model_name: Model to look up.

        Returns:
            Mapping with ``name``, ``digest`` and ``family``.

        Raises:
            RuntimeError: If the server does not have that model.
        """
        if self._model_cache is None:
            self.list_model_names()
        info = (self._model_cache or {}).get(model_name)
        if info is None:
            raise RuntimeError(f"model not found on Ollama server: {model_name}")
        return {
            "name": model_name,
            "digest": info["digest"],
            "family": info.get("details", {}).get("family", "unknown"),
        }

    def assert_model_families_differ(self) -> None:
        """Verifies the two role models are different families, per server metadata.

        Runs once, on the first generation call. Distinct names are not sufficient:
        two tags can share a family or resolve to the same digest, and either case
        means the reviewing model is effectively certifying its own output.

        Raises:
            RuntimeError: If the two models share a family or a digest.
        """
        if self._families_checked:
            return
        draft = self.model_details(self.config.gold_draft_model)
        review = self.model_details(self.config.groundedness_model)
        if draft["digest"] == review["digest"]:
            raise RuntimeError(
                f"gold_draft_model '{draft['name']}' and groundedness_model "
                f"'{review['name']}' resolve to the same digest {draft['digest']}; "
                f"Section 5.5/6 forbids a model certifying its own answer."
            )
        if draft["family"] == review["family"] and draft["family"] != "unknown":
            raise RuntimeError(
                f"gold_draft_model '{draft['name']}' and groundedness_model "
                f"'{review['name']}' are both family '{draft['family']}'; "
                f"Section 5.5/6 requires different model families."
            )
        self._families_checked = True

    def _generate(self, model: str, prompt: str) -> dict[str, Any]:
        """Runs one completion and returns it with its provenance block.

        Args:
            model: Model name to run.
            prompt: Full prompt text.

        Returns:
            Mapping with ``text`` (the stripped completion) and ``model`` (the
            provenance block, including the prompt hash).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": self.config.temperature},
        }
        result = self._request("/api/generate", payload, method="POST")
        details = self.model_details(model)
        return {
            "text": result.get("response", "").strip(),
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
        return self._generate(self.config.gold_draft_model, prompt)

    def groundedness_check(
        self, claim: str, evidence_text: str
    ) -> tuple[str, dict[str, Any]]:
        """Checks whether the evidence supports one claim.

        Always uses ``groundedness_model``. Anything that is not a clear yes or no
        is recorded as ``"partial"`` rather than forced to one side: the verdict
        feeds a human review worksheet, and an ambiguous answer is a real signal
        there.

        The reviewer's own provenance block is returned rather than discarded, so
        the groundedness report can record which weights and which prompt produced
        the verdict. A report that names only the drafting model documents half of
        the integrity claim.

        Args:
            claim: One sentence from the drafted gold answer.
            evidence_text: The evidence block the claim must be supported by.

        Returns:
            Tuple ``(verdict, result)`` where verdict is ``"yes"``, ``"no"`` or
            ``"partial"``, and result carries ``text`` and ``model``.
        """
        prompt = (
            "You are a strict fact-checker. You will be given EVIDENCE (raw log lines) "
            "and a CLAIM. Decide whether the evidence supports the claim.\n"
            "Answer with exactly one word: yes, no, or partial.\n\n"
            f"EVIDENCE:\n{evidence_text}\n\nCLAIM:\n{claim}\n\nAnswer:"
        )
        result = self._generate(self.config.groundedness_model, prompt)
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

        The prompt carries the table shape, the dataset key and the question, and
        nothing else. It does not carry the gold answer, the literal the generator
        searched for, or the cited line numbers: a model shown the answer would be
        asked to reproduce it rather than to derive it, and the check would confirm
        only that it can copy.

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
            "- Match text case-insensitively unless the question demands otherwise.\n"
            "- Return exactly one row and one column: the answer itself.\n"
            "- For a counting question return the count. For a yes/no question return "
            "a boolean. For a question asking for a line, return that line's text.\n"
            "- Output only the SQL statement. No prose, no markdown fence, no semicolon."
        )
        result = self._generate(self.config.sql_model, prompt)
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


def check_server(config: OllamaConfig) -> int:
    """Verifies connectivity, the required models, and the family separation.

    A missing model and an unreachable server return different codes because they
    call for different fixes: one is an ``ollama pull`` on a server this project
    does not own, the other is a network or address problem.

    Args:
        config: Server address and the two role models.

    Returns:
        ``0`` everything present and the families differ, ``1`` a required model is
        missing, ``2`` the server is unreachable, ``3`` the two role models share a
        family or a digest.
    """
    client = OllamaClient(config)

    print(f"Connecting to Ollama server: {config.base_url}")
    try:
        available = client.list_model_names()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "Note: this infrastructure is outside this project's control "
            "(Section 5.5); if the problem persists, check the server address/port "
            "or network reachability.",
            file=sys.stderr,
        )
        return 2

    print(f"Models on server ({len(available)}): {sorted(available)}")

    missing = [name for name in config.required_model_names if name not in available]
    if missing:
        print(
            f"ERROR: required model(s) missing from server: {missing}", file=sys.stderr
        )
        return 1

    try:
        client.assert_model_families_differ()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3

    print("OK: connection established and all required models are present.")
    for name, role in config.role_models:
        details = client.model_details(name)
        print(
            f"  - {name:<24} role={role:<20} family={details['family']:<12} "
            f"digest={details['digest'][:19]}"
        )
    return 0
