# datasetgen container - Section 5.3
# Deterministic/semantic/hard question generation, groundedness and schema
# validation, and output file creation all happen inside this container.
#
# Build context is the repo root, because that is where the Python project lives:
# main.py plus config/ src/ analysis/ scripts/ tests/. The container's /app
# therefore mirrors the repo one-to-one, which is what lets a command documented in
# the README run identically on the host and inside the container.
#
# Base image is pinned to a minor tag rather than a digest. Digest pinning needs a
# registry pull to resolve and would have to be refreshed by hand on every base
# update; the tag plus the pinned requirements.txt is what this project verifies.
FROM python:3.11-slim

# Runs as a non-root user. Nothing here needs privileges: the corpus is mounted
# read-only and the only writable path is /output, whose ownership is set below.
RUN groupadd --gid 1000 datasetgen \
    && useradd --uid 1000 --gid 1000 --create-home datasetgen \
    && mkdir -p /output \
    && chown -R datasetgen:datasetgen /output

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py generate.py validate.py ./
COPY config/ ./config/
COPY src/ ./src/
COPY analysis/ ./analysis/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

USER datasetgen

# Default command: print a help message. Real pipeline commands are supplied via
# `command:` in compose.yml or through `docker compose exec`.
CMD ["python3", "-c", "print('datasetgen ready. Example: docker compose exec datasetgen python3 main.py --command check-ollama')"]
