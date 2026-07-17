#!/usr/bin/env python3
"""
ollama_client.py

Shared client for the remote Ollama server (Section 5.5), used by
layer2_semantic.py (gold draft) and layer3_hard.py (gold draft +
groundedness check). Retry/backoff mirrors check_ollama.py. Concurrency is
capped with a semaphore per scale_config.yaml's
concurrency.max_parallel_model_calls (Section 3.2).

Per the Scientific Integrity Rule (Section 5.5/6), the drafting model
(GOLD_DRAFT_MODEL) and the reviewing model (GROUNDEDNESS_MODEL) must always
be from different families; this module hard-codes that separation as two
distinct functions (draft / groundedness_check) rather than a single
parameterized call, so the two roles can never accidentally collapse into
one model.
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.15.33.66:11435")
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0

GOLD_DRAFT_MODEL = "nemotron-3-nano:30b"
GROUNDEDNESS_MODEL = "gpt-oss:20b"


class OllamaClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, max_parallel_calls=4,
                 max_retries=MAX_RETRIES, backoff_base_seconds=BACKOFF_BASE_SECONDS):
        if GOLD_DRAFT_MODEL == GROUNDEDNESS_MODEL:
            raise RuntimeError("gold-draft and groundedness models must not be identical")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._semaphore = threading.Semaphore(max_parallel_calls)
        self._digest_cache = None

    def _request(self, path, payload=None, method="GET"):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_err = None
        with self._semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    req = urllib.request.Request(
                        url, data=data, method=method,
                        headers={"User-Agent": "logrouter-datasetgen/ollama_client",
                                 "Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except Exception as e:  # noqa: BLE001 - broad catch, needed for retry
                    last_err = e
                    if attempt < self.max_retries:
                        wait = self.backoff_base_seconds * (2 ** (attempt - 1))
                        print(f"  [retry {attempt}/{self.max_retries}] {method} {url} -> {e} ; "
                              f"waiting {wait:.1f}s", file=sys.stderr)
                        time.sleep(wait)
        raise RuntimeError(f"Ollama call failed ({self.max_retries} attempts): {method} {url} :: {last_err}")

    def _model_details(self, model_name):
        if self._digest_cache is None:
            payload = self._request("/api/tags")
            self._digest_cache = {m["name"]: m for m in payload.get("models", [])}
        info = self._digest_cache.get(model_name)
        if info is None:
            raise RuntimeError(f"model not found on Ollama server: {model_name}")
        return {
            "digest": info["digest"],
            "family": info.get("details", {}).get("family", "unknown"),
        }

    def _generate(self, model, prompt, temperature=0.0):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,  # Section 7.3: disable thinking mode when the model supports it
            "options": {"temperature": temperature},
        }
        result = self._request("/api/generate", payload, method="POST")
        details = self._model_details(model)
        return {
            "text": result.get("response", "").strip(),
            "model": {
                "name": model,
                "family": details["family"],
                "digest": details["digest"],
                "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            },
            "raw": result,
        }

    def draft(self, prompt, temperature=0.0):
        """Gold-answer draft. Always GOLD_DRAFT_MODEL (Section 5.5)."""
        return self._generate(GOLD_DRAFT_MODEL, prompt, temperature=temperature)

    def groundedness_check(self, claim, evidence_text):
        """
        Checks whether `evidence_text` supports `claim`. Always GROUNDEDNESS_MODEL,
        a different family from GOLD_DRAFT_MODEL (Section 5.5/6). Returns
        (verdict, raw_result) where verdict is one of "yes" | "no" | "partial".
        """
        prompt = (
            "You are a strict fact-checker. You will be given EVIDENCE (raw log lines) "
            "and a CLAIM. Decide whether the evidence supports the claim.\n"
            "Answer with exactly one word: yes, no, or partial.\n\n"
            f"EVIDENCE:\n{evidence_text}\n\nCLAIM:\n{claim}\n\nAnswer:"
        )
        result = self._generate(GROUNDEDNESS_MODEL, prompt, temperature=0.0)
        answer = result["text"].strip().lower()
        if answer.startswith("yes"):
            verdict = "yes"
        elif answer.startswith("no"):
            verdict = "no"
        else:
            verdict = "partial"
        return verdict, result
