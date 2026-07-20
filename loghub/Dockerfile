# loghub container - Section 5.2
# Fetches the LogHub 2k sample files pinned in corpus_manifest.json into the
# shared /data/loghub volume, then stays up (does not exit) so the service
# remains a long-lived member of datasetgen-net. Readiness is reported via
# HEALTHCHECK rather than process exit.
FROM python:3.11-slim

WORKDIR /app

COPY corpus_manifest.json fetch_corpus.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Standard library only (urllib, hashlib, json); no extra dependencies.
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=60 \
    CMD test -f /data/loghub/.ready || exit 1

ENTRYPOINT ["./entrypoint.sh"]
