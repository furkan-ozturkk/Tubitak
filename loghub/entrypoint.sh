#!/bin/bash
# loghub container entrypoint - Section 5.2
# This container is now both the corpus source AND the Postgres server the
# corpus is queried through:
#   1) Start Postgres using the base image's own entrypoint logic (handles
#      initdb-if-needed, permission drop to the postgres user, and running
#      any /docker-entrypoint-initdb.d/ hook scripts on first init only).
#   2) Wait until it actually accepts connections.
#   3) Make sure pgvector is enabled (idempotent).
#   4) Fetch the pinned LogHub corpus (idempotent -- fetch_corpus.py skips
#      re-download if already on disk) and load it into the `lines` table.
#   5) Drop the readiness marker HEALTHCHECK looks for.
#   6) Keep Postgres as the container's main process.
set -e

: "${POSTGRES_USER:=postgres}"
: "${POSTGRES_DB:=postgres}"

docker-entrypoint.sh postgres &
PG_PID=$!

echo "Waiting for Postgres to accept connections..."
until pg_isready -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; do
    sleep 1
done

psql -h localhost -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"

python3 fetch_corpus.py --manifest /app/corpus_manifest.json --output-dir /data/loghub --load-postgres
touch /data/loghub/.ready

# Keep Postgres as the container's main process so it stays up and signals
# (stop/restart) are handled by the same process Docker is tracking.
wait "$PG_PID"
