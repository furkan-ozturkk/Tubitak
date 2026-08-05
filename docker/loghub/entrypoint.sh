#!/bin/bash
# loghub container entrypoint - Section 5.2
#
# Dispatches on a subcommand so the documented one-shot checks actually reach the
# fetch script. Previously this script ignored "$@" entirely, which meant
# `docker compose run --rm loghub --verify-only` silently started a full Postgres
# + fetch + load cycle instead of verifying anything -- and, sharing the pgdata
# volume with the running service, could fail on a second Postgres startup.
#
#   serve   (default) start Postgres, enable pgvector, fetch, load, stay up
#   verify            checksum-verify the files already on disk, then exit
#   fetch             fetch/verify the corpus without touching Postgres, then exit
#   <other>           exec the given command, for debugging
#
# Usage:
#   docker compose -f docker/compose.yml up loghub
#   docker compose -f docker/compose.yml run --rm --no-deps loghub verify
#   docker compose -f docker/compose.yml run --rm --no-deps loghub fetch
set -euo pipefail

: "${POSTGRES_USER:=postgres}"
: "${POSTGRES_DB:=postgres}"

MANIFEST=/app/corpus_manifest.json
CORPUS_DIR=/data/loghub
READY_MARKER="$CORPUS_DIR/.ready"

command=${1:-serve}

case "$command" in
  verify)
    # Never writes the lock: in this mode a missing lock is a hard error, because
    # writing one would bless whatever happens to be on disk as the truth.
    exec python3 fetch_corpus.py --manifest "$MANIFEST" --output-dir "$CORPUS_DIR" --verify-only
    ;;

  fetch)
    exec python3 fetch_corpus.py --manifest "$MANIFEST" --output-dir "$CORPUS_DIR"
    ;;

  serve)
    # 1) Start Postgres via the base image's own entrypoint, which handles
    #    initdb-if-needed, the permission drop to the postgres user, and any
    #    /docker-entrypoint-initdb.d/ hooks on first init only.
    # 2) Wait until it accepts connections.
    # 3) Ensure pgvector is enabled (idempotent).
    # 4) Fetch the pinned corpus (idempotent) and load it into the `lines` table.
    # 5) Drop the readiness marker HEALTHCHECK looks for. It is removed first so a
    #    restart cannot report ready on a stale marker while the fetch is still
    #    running.
    # 6) Keep Postgres as the container's main process so stop/restart signals
    #    reach the process Docker is tracking.
    rm -f "$READY_MARKER"

    docker-entrypoint.sh postgres &
    PG_PID=$!

    echo "Waiting for Postgres to accept connections..."
    until pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
      sleep 1
    done

    psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
      -c "CREATE EXTENSION IF NOT EXISTS vector;"

    python3 fetch_corpus.py \
      --manifest "$MANIFEST" \
      --output-dir "$CORPUS_DIR" \
      --load-postgres

    touch "$READY_MARKER"

    wait "$PG_PID"
    ;;

  *)
    exec "$@"
    ;;
esac
