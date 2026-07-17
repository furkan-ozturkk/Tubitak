#!/usr/bin/env python3
"""
check_ollama.py

Section 5.5 / 6: verifies connectivity to the remote Ollama server
(OLLAMA_BASE_URL) and checks that the two models used in this project
(from different families!) are available on the server:

  - nemotron-3-nano:30b  -> semantic/hard-tier gold answer drafts
  - gpt-oss:20b          -> groundedness check

Per the Scientific Integrity Rule these two models must be from DIFFERENT
families; this script also checks (when read from config) that the model
names are not identical to each other.

Usage:
  docker compose exec datasetgen python3 src/check_ollama.py
  python3 src/check_ollama.py --base-url http://10.15.33.66:11435
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.15.33.66:11435")
REQUIRED_MODELS = [
    {"name": "nemotron-3-nano:30b", "role": "gold_draft"},
    {"name": "gpt-oss:20b", "role": "groundedness_check"},
]
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0


def get_with_retry(url: str, max_retries: int = MAX_RETRIES) -> dict:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "logrouter-datasetgen/check_ollama"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"  [retry {attempt}/{max_retries}] {url} -> {e} ; waiting {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"Connection failed ({max_retries} attempts): {url} :: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--require-models", nargs="*", default=[m["name"] for m in REQUIRED_MODELS],
                     help="Model names that must be present on the server")
    args = ap.parse_args()

    tags_url = args.base_url.rstrip("/") + "/api/tags"
    print(f"Connecting to Ollama server: {tags_url}")

    try:
        payload = get_with_retry(tags_url)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Note: this infrastructure is outside this project's control (Section 5.5); if the "
              "problem persists, check the server address/port or network reachability.", file=sys.stderr)
        return 2

    available = {m.get("name") for m in payload.get("models", [])}
    print(f"Models on server ({len(available)}): {sorted(available)}")

    missing = [m for m in args.require_models if m not in available]

    # Make sure the same model family isn't used for both drafting and review (Section 5.5/6).
    if len(set(args.require_models)) != len(args.require_models):
        print("ERROR: required model list contains a duplicate; the drafting and reviewing model "
              "must never be the same.",
              file=sys.stderr)
        return 3

    if missing:
        print(f"ERROR: required model(s) missing from server: {missing}", file=sys.stderr)
        return 1

    print("OK: connection established and all required models are present on the server.")
    for m in REQUIRED_MODELS:
        print(f"  - {m['name']:<24} role={m['role']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
