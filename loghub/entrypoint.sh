#!/bin/sh
# loghub container entrypoint - Section 5.2
# Fetches the pinned LogHub corpus once, drops a readiness marker for the
# healthcheck, then keeps the container running so it stays available on
# datasetgen-net instead of exiting after the one-shot fetch.
set -e

python3 fetch_corpus.py --manifest /app/corpus_manifest.json --output-dir /data/loghub
touch /data/loghub/.ready

exec tail -f /dev/null
