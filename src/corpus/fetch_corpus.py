"""Fetches and verifies the pinned LogHub corpus (Section 4.1).

Runs inside the loghub container, not the datasetgen one, and therefore imports
nothing from the rest of ``src``. That container is a Postgres image with Python
added for this one script; pulling the question-generation app into it would give
the corpus fetcher a dependency set it has no use for and cannot install.
``tests/test_corpus_isolation.py`` asserts the boundary rather than trusting it.

The Scientific Integrity Rule this implements: data is never downloaded by hand and
dropped into a folder. Every dataset comes from the single commit pinned in
``corpus_manifest.json``, and every file's SHA-256 is recorded in a lock file that
later runs are checked against.

What the lock can and cannot establish is the crux, and the earlier version
overstated it. A lock written from whatever happened to be on disk proves nothing:
if the volume already held files with the right names, a normal startup accepted
them without a download and then blessed their digests as the truth. Two rules fix
that.

``--verify-only`` never writes a lock. Without one there is nothing to verify
against, so a missing lock is a hard error rather than an invitation to create one
from the current contents.

A file already on disk is only accepted when the lock already covers it. On a first
run with no lock, every dataset is fetched from the pinned commit, which is the only
source whose digests may become the lock's. ``--trust-existing`` exists for the case
where a lock has to be established from an offline copy, and it says so in its name
and in the lock's ``locked_from`` field rather than being the silent default.

Digests belong in version control. ``--lock-file`` defaults into the output volume,
where the lock is a runtime artefact; passing the repo's own
``corpus_manifest.lock.json`` makes it a reviewed one.

With ``--load-postgres``, a successful pass also loads every dataset into a
``lines(dataset, line_number, text)`` table so datasetgen can query the corpus with
SQL. The table carries a uniqueness constraint on ``(dataset, line_number)``, since
a duplicated row would make every count computed from it wrong while every
individual lookup still looked correct.

Usage:
  python3 fetch_corpus.py --manifest corpus_manifest.json --output-dir /data/loghub
  python3 fetch_corpus.py --verify-only --output-dir /data/loghub
  python3 fetch_corpus.py --output-dir /data/loghub --load-postgres
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).parent / "corpus_manifest.json"
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MIN_PLAUSIBLE_LINES = 3


def corpus_lines(data: bytes) -> list[str]:
    """Splits corpus bytes into lines using the shared line contract.

    Mirrors ``src.data.corpus_loader.lines_from_bytes`` without importing it —
    this script runs in the loghub container, which has no ``src`` package.
    The contract is ``str.splitlines()`` so that CRLF endings (which the LogHub
    2k files use) never leak a ``\\r`` into a stored line, its hash, or the
    evaluation harness that re-reads the same files. ``tests/test_line_contract.py``
    asserts the two implementations stay identical.

    Args:
        data: Raw bytes of a ``*_2k.log`` file.

    Returns:
        The file's lines, without terminators. Index ``i`` is line ``i + 1``.
    """
    return data.decode("utf-8", errors="replace").splitlines()


def sha256_of_bytes(data: bytes) -> str:
    """Returns the hex SHA-256 of a byte string.

    Args:
        data: Bytes to hash.

    Returns:
        The hex digest, without a prefix.
    """
    return hashlib.sha256(data).hexdigest()


def fetch_with_retry(url: str, max_retries: int = MAX_RETRIES) -> bytes:
    """Downloads a URL, retrying with exponential backoff.

    The response is read with an explicit cap rather than to completion. These files
    are a few hundred kilobytes; anything approaching the cap means the URL is not
    what the manifest thinks it is, and reading it into memory first would be the
    wrong way to find out.

    Args:
        url: Raw file URL at the pinned commit.
        max_retries: Attempts before giving up.

    Returns:
        The file's bytes.

    Raises:
        RuntimeError: If every attempt failed, or the response exceeded the cap.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "logrouter-datasetgen/fetch_corpus"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(
                        url, response.status, "non-200", response.headers, None
                    )
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
                if len(data) > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        f"response exceeded {MAX_DOWNLOAD_BYTES} bytes; "
                        f"the manifest URL is probably wrong"
                    )
                return data
        except Exception as error:
            last_err = error
            if attempt < max_retries:
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                print(
                    f"  [retry {attempt}/{max_retries}] {url} -> {error} ; "
                    f"waiting {wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
    raise RuntimeError(
        f"Fetch failed (after {max_retries} attempts): {url} :: {last_err}"
    )


def load_manifest(path: Path) -> dict:
    """Reads the pinned manifest.

    Args:
        path: ``corpus_manifest.json`` path.

    Returns:
        The parsed manifest.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_lock(path: Path) -> dict | None:
    """Reads the lock file if it exists.

    Args:
        path: Lock file path.

    Returns:
        The parsed lock, or ``None`` when no lock exists. ``None`` is returned rather
        than an empty lock so callers must decide what a missing lock means; treating
        it as an empty one is what allowed unverified files to be blessed.
    """
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_lock(path: Path, lock: dict) -> None:
    """Writes the lock file atomically.

    A lock truncated by a process killed mid-write would fail every later run's
    comparison, and the obvious fix at that point looks like deleting it, which
    discards the digests being protected.

    Args:
        path: Lock file path.
        lock: Lock contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".lock.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            json.dump(lock, temp_file, ensure_ascii=False, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def resolve_local_path(output_dir: Path, local_filename: str) -> Path:
    """Resolves a manifest filename inside the output directory.

    The manifest is data, and a custom one could name ``../../etc/passwd`` or an
    absolute path. Resolving and then checking containment keeps a manifest from
    directing a write outside the corpus volume.

    Args:
        output_dir: Corpus directory.
        local_filename: ``local_filename`` from a manifest entry.

    Returns:
        The resolved path inside ``output_dir``.

    Raises:
        ValueError: If the name escapes ``output_dir``.
    """
    base = output_dir.resolve()
    candidate = (base / local_filename).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(
            f"manifest local_filename '{local_filename}' resolves outside the output "
            f"directory ({candidate}); refusing to write there"
        )
    return candidate


def write_corpus_file(path: Path, data: bytes) -> None:
    """Writes a corpus file atomically.

    ``tempfile.mkstemp`` always creates its file mode ``0600`` (owner-only), a
    security default that survives ``os.replace``. This container's corpus is
    read by a different container running as a different, non-root user
    (``datasetgen``), so the mode is widened to world-readable before the
    rename; otherwise every read of a freshly-fetched file fails with
    ``PermissionError`` for anyone but the user that fetched it.

    Args:
        path: Destination file.
        data: File contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def load_into_postgres(output_dir: Path, manifest: dict, pg_config: dict) -> None:
    """Loads every fetched dataset's lines into the ``lines`` table.

    Idempotent per dataset: the dataset's rows are deleted and reinserted inside one
    transaction, so a container restart cannot leave a half-loaded or doubled corpus.
    ``UNIQUE (dataset, line_number)`` makes a duplicate a failed insert rather than a
    silently inflated count.

    Args:
        output_dir: Corpus directory.
        manifest: The parsed manifest.
        pg_config: Connection parameters.
    """
    import psycopg2
    from psycopg2.extras import execute_values

    connection = psycopg2.connect(
        host=pg_config["host"],
        port=pg_config["port"],
        dbname=pg_config["dbname"],
        user=pg_config["user"],
        password=pg_config["password"],
    )
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS lines (
                        id BIGSERIAL PRIMARY KEY,
                        dataset TEXT NOT NULL,
                        line_number INTEGER NOT NULL,
                        text TEXT NOT NULL
                    );
                    """)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_lines_dataset ON lines(dataset);"
                )
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_lines_dataset_line "
                    "ON lines(dataset, line_number);"
                )

                for dataset in manifest["datasets"]:
                    dataset_key = dataset["name"].lower()
                    local_path = resolve_local_path(
                        output_dir, dataset["local_filename"]
                    )
                    file_lines = corpus_lines(local_path.read_bytes())

                    cursor.execute(
                        "DELETE FROM lines WHERE dataset = %s", (dataset_key,)
                    )
                    rows = [
                        (dataset_key, index + 1, line)
                        for index, line in enumerate(file_lines)
                    ]
                    execute_values(
                        cursor,
                        "INSERT INTO lines (dataset, line_number, text) VALUES %s",
                        rows,
                    )
                    print(
                        f"  [postgres] loaded {len(rows)} lines for dataset={dataset_key}"
                    )
    finally:
        connection.close()


def pg_config_from_env() -> dict:
    """Reads Postgres connection parameters from the environment.

    Returns:
        Connection parameters for ``load_into_postgres``.
    """
    return {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "postgres"),
        "user": os.environ.get("POSTGRES_USER", "postgres"),
        "password": os.environ.get("POSTGRES_PASSWORD", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    """Builds this script's argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=None,
        help="Default: <output-dir>/corpus_manifest.lock.json. Point this at a lock "
        "committed to the repository to make the approved digests reviewable.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Make no network calls and never write the lock; compare the files in "
        "output-dir against an existing lock. Fails if there is no lock, because "
        "there is then nothing to verify against.",
    )
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Re-download even if the file is already on disk (the lock is still verified).",
    )
    parser.add_argument(
        "--trust-existing",
        action="store_true",
        help="When no lock exists, allow files already in output-dir to establish it "
        "instead of fetching from the pinned commit. Recorded in the lock's "
        "locked_from field; without it a first run always downloads.",
    )
    parser.add_argument(
        "--load-postgres",
        action="store_true",
        help="After a successful fetch/verify pass, load all datasets into Postgres "
        "(connection read from PGHOST/PGPORT/POSTGRES_DB/POSTGRES_USER/POSTGRES_PASSWORD).",
    )
    return parser


def main() -> int:
    """Fetches or verifies the corpus, then optionally loads it into Postgres.

    Returns:
        ``0`` success, ``1`` a dataset failed, ``2`` the run was refused before any
        dataset was processed (missing lock in verify mode, or a commit that
        disagrees with the lock).
    """
    args = build_parser().parse_args()

    manifest = load_manifest(args.manifest)
    source = manifest["source"]
    owner, repo, commit = source["owner"], source["repo"], source["pinned_commit"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.lock_file or (args.output_dir / "corpus_manifest.lock.json")
    lock = load_lock(lock_path)

    if lock is None:
        if args.verify_only:
            print(
                f"ERROR: --verify-only needs an existing lock, and none was found at "
                f"{lock_path}. Writing one now would record whatever is currently on "
                f"disk as the approved corpus, which is what verification is meant to "
                f"rule out. Run a normal fetch first.",
                file=sys.stderr,
            )
            return 2
        lock = {
            "pinned_commit": commit,
            "locked_from": "existing-files" if args.trust_existing else "pinned-commit",
            "entries": {},
        }
        print(f"No lock at {lock_path}; establishing one from {lock['locked_from']}.")

    if lock.get("pinned_commit") not in (None, commit):
        print(
            f"ERROR: pinned commit in lock file ({lock['pinned_commit']}) differs from "
            f"manifest ({commit}). The corpus must not change silently.",
            file=sys.stderr,
        )
        return 2
    lock["pinned_commit"] = commit
    lock.setdefault("locked_from", "pinned-commit")
    lock.setdefault("entries", {})

    succeeded, failed, quota_unmet = [], [], []

    for dataset in manifest["datasets"]:
        name = dataset["name"]
        try:
            local_path = resolve_local_path(args.output_dir, dataset["local_filename"])
        except ValueError as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            failed.append(name)
            continue

        raw_url = source["raw_base_url"].format(
            owner=owner, repo=repo, pinned_commit=commit, raw_path=dataset["raw_path"]
        )
        locked_entry = lock["entries"].get(name)

        print(f"[{name}] target: {local_path.name}")

        may_read_from_disk = local_path.exists() and (
            locked_entry is not None or args.trust_existing or args.verify_only
        )

        if args.verify_only:
            if not local_path.exists():
                print(
                    f"  ERROR: --verify-only requested but file is missing: {local_path}",
                    file=sys.stderr,
                )
                failed.append(name)
                continue
            if locked_entry is None:
                print(
                    f"  ERROR: the lock has no entry for {name}; it cannot be verified.",
                    file=sys.stderr,
                )
                failed.append(name)
                continue
            data = local_path.read_bytes()
            source_desc = "read from disk (verify-only)"
        elif may_read_from_disk and not args.force_refetch:
            data = local_path.read_bytes()
            source_desc = "read from disk (already present)"
        else:
            try:
                data = fetch_with_retry(raw_url)
            except RuntimeError as error:
                print(f"  ERROR: {error}", file=sys.stderr)
                failed.append(name)
                continue
            source_desc = "freshly downloaded"

        digest = sha256_of_bytes(data)
        line_count = len(corpus_lines(data))

        if line_count < MIN_PLAUSIBLE_LINES:
            print(
                f"  WARNING: {name} has fewer lines than expected ({line_count}); "
                f"marked as quota_unmet."
            )
            quota_unmet.append(name)

        if locked_entry is not None:
            if locked_entry["sha256"] != digest:
                print(
                    f"  ERROR: checksum mismatch! lock={locked_entry['sha256']} "
                    f"new={digest}\n"
                    f"  The corpus cannot have changed while the commit is pinned -- "
                    f"the fetch or the lock file may be corrupted.",
                    file=sys.stderr,
                )
                failed.append(name)
                continue
            print(
                f"  OK ({source_desc}): {line_count} lines, sha256={digest[:16]}... "
                f"(matches lock)"
            )
        else:
            lock["entries"][name] = {
                "raw_path": dataset["raw_path"],
                "local_filename": dataset["local_filename"],
                "sha256": digest,
                "line_count": line_count,
                "byte_size": len(data),
            }
            print(
                f"  OK ({source_desc}): {line_count} lines, sha256={digest[:16]}... "
                f"(saved to lock)"
            )

        if not args.verify_only and (
            not local_path.exists() or source_desc == "freshly downloaded"
        ):
            write_corpus_file(local_path, data)

        succeeded.append(name)

    if not args.verify_only:
        save_lock(lock_path, lock)

    print("\n=== SUMMARY ===")
    print(f"Succeeded: {len(succeeded)}/{len(manifest['datasets'])} -> {succeeded}")
    if quota_unmet:
        print(f"quota_unmet (suspicious line count): {quota_unmet}")
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1

    print(f"Lock file: {lock_path} (locked_from={lock['locked_from']})")

    if args.load_postgres:
        print("\n=== Loading into Postgres ===")
        load_into_postgres(args.output_dir, manifest, pg_config_from_env())

    return 0


if __name__ == "__main__":
    sys.exit(main())
