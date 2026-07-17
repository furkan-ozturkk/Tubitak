# LogRouter Evaluation Dataset - Production Architecture

This repository contains the independent, scalable architecture that produces
the LogRouter black-box evaluation question-answer dataset described in
`staj.pdf`. It has no connection to LogRouter's source code or infrastructure
(Section 1.1).

## Status: Phase 0 (Preparation) complete

- [x] LogHub source pinned (`loghub/corpus_manifest.json`) and the fetch
      mechanism written (`loghub/fetch_corpus.py`).
- [x] Two-container Docker environment (`docker-compose.yml`); the `loghub`
      service stays running (reports readiness via HEALTHCHECK instead of
      exiting) so it remains reachable on the shared network.
- [x] Ollama connectivity/model verification script (`datasetgen/src/check_ollama.py`).
- [x] Question schema + automated validator (`datasetgen/src/schema/`).

Next up: **Phase 1 - Pilot** (~100 questions, Section 3.1).

## Directory layout

```
.
├── docker-compose.yml       # two containers, one network, one external connection
├── .env.example              # OLLAMA_BASE_URL
├── scale_config.yaml         # single configuration source for Phase 2/3
├── loghub/
│   ├── Dockerfile
│   ├── entrypoint.sh         # fetch once, then stay alive (no exit)
│   ├── corpus_manifest.json  # pinned commit + 10 datasets
│   └── fetch_corpus.py       # fetch + checksum + lock verification
├── datasetgen/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── check_ollama.py
│       └── schema/
│           ├── question_schema.json   # Section 6 schema (single source of truth)
│           └── validate_schema.py     # schema + cross-record + groundedness validation
└── output/                   # dataset/report output exported to the host
```

## Running it (Phase 0 verification)

```bash
cp .env.example .env   # edit OLLAMA_BASE_URL if needed
docker compose up --build
# loghub fetches the corpus, reports healthy, and keeps running;
# datasetgen starts once loghub is healthy and stays up (tail -f).

# Verify the Ollama connection and the required models:
docker compose exec datasetgen python3 src/check_ollama.py

# Confirm the corpus was fetched correctly:
docker compose exec datasetgen python3 /app/../loghub/fetch_corpus.py \
    --verify-only --output-dir /data/loghub \
    --manifest /app/../loghub/corpus_manifest.json
# (fetch_corpus.py lives in the loghub image, so in practice you can also run:
#  docker compose run --rm loghub --verify-only)

# Test the schema validator against a sample question file:
docker compose exec datasetgen python3 src/schema/validate_schema.py \
    --questions /output/sample_questions.json \
    --corpus-dir /data/loghub \
    --manifest /data/loghub/../../loghub/corpus_manifest.json \
    --report /output/validation_report.json
```

## Scientific Integrity Rules (summary)

1. **Data source**: LogHub data is fetched only via `fetch_corpus.py`, from a
   pinned commit; no manual downloads (Section 4.1).
2. **Generator != Reviewer**: the model that drafts the gold answer
   (`nemotron-3-nano:30b`) and the model that performs the groundedness check
   (`gpt-oss:20b`) must always be from different families (Section 5.5/6).
3. **Access boundary**: LogRouter's source code, infrastructure, or internal
   architecture are never used or referenced in this project in any way
   (Section 1.1/3.2).
