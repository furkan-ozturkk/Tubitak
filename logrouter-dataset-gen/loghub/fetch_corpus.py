#!/usr/bin/env python3
"""
fetch_corpus.py

LogRouter Evaluation Dataset project - Section 4.1 Scientific Integrity Rule:
Data is never downloaded by hand and dropped into an arbitrary folder. LogHub
datasets are fetched from a single commit pinned in corpus_manifest.json.

This script:
  1. For each dataset in corpus_manifest.json, fetches the *_2k.log file via
     raw.githubusercontent.com from the pinned commit (with retry + backoff).
  2. Computes the SHA-256 checksum of every downloaded file.
  3. Creates corpus_manifest.lock.json if it does not exist yet (first run =
     "locking"); if it exists, compares it byte-for-byte against the new
     result. On mismatch it raises an ERROR and stops (prevents the corpus
     from silently changing).
  4. With --verify-only, only checks the checksum of files already on disk;
     makes no network calls.

Usage:
  python3 fetch_corpus.py --manifest corpus_manifest.json --output-dir /data/loghub
  python3 fetch_corpus.py --verify-only --output-dir /data/loghub
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).parent / "corpus_manifest.json"
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_with_retry(url: str, max_retries: int = MAX_RETRIES) -> bytes:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "logrouter-datasetgen/fetch_corpus"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(url, resp.status, "non-200", resp.headers, None)
                return resp.read()
        except Exception as e:  # noqa: BLE001 - broad catch, needed for retry
            last_err = e
            if attempt < max_retries:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"  [retry {attempt}/{max_retries}] {url} -> {e} ; waiting {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"Fetch failed (after {max_retries} attempts): {url} :: {last_err}")


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lock(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"pinned_commit": None, "entries": {}}


def save_lock(path: Path, lock: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lock, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--lock-file", type=Path, default=None,
                     help="Default: <output-dir>/corpus_manifest.lock.json")
    ap.add_argument("--verify-only", action="store_true",
                     help="Make no network calls; only compare files in output-dir against the lock.")
    ap.add_argument("--force-refetch", action="store_true",
                     help="Re-download even if the file is already on disk (lock is still verified).")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    source = manifest["source"]
    owner, repo, commit = source["owner"], source["repo"], source["pinned_commit"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.lock_file or (args.output_dir / "corpus_manifest.lock.json")
    lock = load_lock(lock_path)

    if lock["pinned_commit"] is not None and lock["pinned_commit"] != commit:
        print(f"ERROR: pinned commit in lock file ({lock['pinned_commit']}) differs from manifest "
              f"({commit}). The corpus must not change silently.", file=sys.stderr)
        return 2
    lock["pinned_commit"] = commit

    ok, failed, quota_unmet = [], [], []

    for ds in manifest["datasets"]:
        name = ds["name"]
        local_path = args.output_dir / ds["local_filename"]
        raw_url = source["raw_base_url"].format(owner=owner, repo=repo, pinned_commit=commit, raw_path=ds["raw_path"])

        print(f"[{name}] target: {local_path.name}")

        if args.verify_only or (local_path.exists() and not args.force_refetch):
            if not local_path.exists():
                print(f"  ERROR: --verify-only requested but file is missing: {local_path}", file=sys.stderr)
                failed.append(name)
                continue
            data = local_path.read_bytes()
            source_desc = "read from disk (verify-only / already present)"
        else:
            try:
                data = fetch_with_retry(raw_url)
            except RuntimeError as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                failed.append(name)
                continue
            source_desc = "freshly downloaded"

        digest = sha256_of_bytes(data)
        line_count = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)

        if line_count < 3:
            # The system reports this explicitly instead of erroring out (Section 3.2).
            print(f"  WARNING: {name} has fewer lines than expected ({line_count}); marked as quota_unmet.")
            quota_unmet.append(name)

        prev = lock["entries"].get(name)
        if prev is not None:
            if prev["sha256"] != digest:
                print(f"  ERROR: checksum mismatch! lock={prev['sha256']} new={digest}\n"
                      f"  The corpus cannot have changed while the commit is pinned -- the fetch or "
                      f"the lock file may be corrupted.",
                      file=sys.stderr)
                failed.append(name)
                continue
            print(f"  OK ({source_desc}): {line_count} lines, sha256={digest[:16]}... (matches lock)")
        else:
            lock["entries"][name] = {
                "raw_path": ds["raw_path"],
                "local_filename": ds["local_filename"],
                "sha256": digest,
                "line_count": line_count,
                "byte_size": len(data),
            }
            print(f"  OK ({source_desc}): {line_count} lines, sha256={digest[:16]}... (saved to lock)")

        if not local_path.exists() or args.force_refetch or not args.verify_only:
            local_path.write_bytes(data)

        ok.append(name)

    save_lock(lock_path, lock)

    print("\n=== SUMMARY ===")
    print(f"Succeeded: {len(ok)}/{len(manifest['datasets'])} -> {ok}")
    if quota_unmet:
        print(f"quota_unmet (suspicious line count): {quota_unmet}")
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1

    print(f"Lock file: {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
