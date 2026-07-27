# loghub container - Section 5.2
# Fetches the LogHub 2k sample files pinned in src/corpus/corpus_manifest.json into
# the shared /data/loghub volume AND serves them from a PostgreSQL (+ pgvector)
# database in the same container, so datasetgen can query the corpus with real
# SQL instead of reading the raw files directly. Based on the official pgvector
# image (Postgres 16 + pgvector precompiled) with Python added on top for the
# fetch script.
#
# Build context is the repo root (see docker/compose.yml), which is why the COPY
# paths below are repo-relative rather than bare filenames.
FROM pgvector/pgvector:pg16

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/corpus/corpus_manifest.json src/corpus/fetch_corpus.py ./
COPY docker/loghub-entrypoint.sh ./
RUN chmod +x loghub-entrypoint.sh

# Ready only once both the corpus has been fetched (.ready marker) and Postgres
# itself is accepting connections.
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=60 \
    CMD test -f /data/loghub/.ready && pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-postgres}" || exit 1

ENTRYPOINT ["./loghub-entrypoint.sh"]
