# LogRouter Evaluation Dataset - Production Architecture

This repository contains the independent, scalable architecture that produces
the LogRouter black-box evaluation question-answer dataset described in
`staj.pdf`. It has no connection to LogRouter's source code or infrastructure
(Section 1.1).

## Status: Faz 1 (Pilot) - Stage 1 (20 questions) complete

- [x] LogHub source pinned (`loghub/corpus_manifest.json`) and the fetch
      mechanism written (`loghub/fetch_corpus.py`).
- [x] Two-container Docker environment (`docker-compose.yml`); the `loghub`
      service stays running (reports readiness via HEALTHCHECK instead of
      exiting) so it remains reachable on the shared network.
- [x] Ollama connectivity/model verification (`main.py check-ollama`).
- [x] Question schema + automated validator (`question_schema.json`, `schema_validator.py`).
- [x] Official output (`output/pilot/questions.json`) is a 20-question starting
      set: 1 count + 1 presence question per LogHub dataset (10 datasets x 2),
      all `difficulty=easy`/`routing_path=sql`. No LLM is involved in
      computing these -- both generation (`question_generators.py`) and
      validation (`schema_validator.py`) run real SQL against loghub's own
      Postgres `lines` table via `pg_client.py`, so `review_status=verified`
      for all 20. 0 schema errors.
- [x] The full first-pass pilot (116 questions across all three difficulty
      tiers) is kept at `output/pilot/questions_full_v1_116.json` and broken
      out per tier in `layer1.json`/`layer2.json`/`layer3.json`. Medium/hard
      questions there are model-drafted and still `review_status=in_review`
      (Section 7.3 step 5) -- they get folded back into the official output
      once a human has reviewed them via `main.py review-export`/`review-apply`,
      growing the 20 upward per the staged-scaling plan (Section 3).
- [x] A separate 28-question mixed-difficulty sample for advisor review is
      available at `output/pilot/mini20/` and `output/pilot/sample_for_review.md`
      (unrelated to the official output above).

## Directory layout

```
.
├── docker-compose.yml       # two containers, one network, one external connection
├── .env.example              # OLLAMA_BASE_URL, POSTGRES_DB/USER/PASSWORD
├── scale_config.yaml         # single configuration source
├── loghub/
│   ├── Dockerfile             # pgvector/pgvector:pg16 + Python (fetch script deps)
│   ├── entrypoint.sh          # start Postgres, enable pgvector, fetch, load, stay up
│   ├── corpus_manifest.json   # pinned commit + 10 datasets
│   └── fetch_corpus.py        # fetch + checksum + lock verification + load_into_postgres()
├── sql_verification/         # optional manual spot-check (see below)
│   └── verify.sql            # the 20 official-output answers, as plain SQL
├── datasetgen/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                # single entry point, one subcommand per operation
│   ├── cli.py                 # every CLI parameter, as one frozen dataclass per subcommand
│   ├── scale_config.py        # scale_config.yaml, typed into RunConfig/DifficultyMix/ConcurrencyConfig
│   ├── question_generators.py # the three question tiers (Section 7): easy/medium/hard
│   ├── loghub_datasets.py     # per-LogHub-dataset literals/hard-group specs (Section 7.1)
│   ├── ollama_client.py       # shared remote-Ollama client (Section 5.5)
│   ├── pg_client.py           # queries loghub's Postgres `lines` table (easy tier + validation)
│   ├── corpus_utils.py        # corpus loading, hashing, dev/test split assignment
│   ├── schema_validator.py    # schema + cross-record + evidence/groundedness validation
│   ├── human_review.py        # human review worksheet export/apply (Section 7.3 step 5)
│   └── question_schema.json   # Section 6 schema (single source of truth)
└── output/
    └── pilot/                # Faz 1 pilot dataset + reports (tracked in git)
```

Every file above sits directly in `datasetgen/` -- no nested `src/`/`schema/`
subfolders; opening the repo shows the real files immediately.

## Running it

```bash
cp .env.example .env   # edit OLLAMA_BASE_URL if needed; set a real POSTGRES_PASSWORD
docker compose up --build
# loghub fetches the corpus, starts Postgres (+pgvector), loads the corpus
# into it, reports healthy, and keeps running; datasetgen starts once loghub
# is healthy and stays up (tail -f).
```

Everything below is one subcommand of `main.py`, run inside the `datasetgen`
container. Defaults match the paths docker-compose.yml mounts, so most of
these work with no extra flags -- pass `--help` on a subcommand to see all
available overrides.

```bash
# Verify the Ollama connection and the required models (Section 5.5/6):
docker compose exec datasetgen python3 main.py check-ollama

# Confirm the corpus was fetched correctly (runs in the loghub image):
docker compose run --rm loghub --verify-only

# Generate the dataset (writes /output/pilot/questions.json by default):
docker compose exec datasetgen python3 main.py generate

# Validate it against the schema (writes /output/pilot/validation_report.json):
docker compose exec datasetgen python3 main.py validate

# Export in_review (medium/hard) records for human review, then apply decisions:
docker compose exec datasetgen python3 main.py review-export
#  ... a human fills in the "decision" column (accept | edit | reject) ...
docker compose exec datasetgen python3 main.py review-apply
```

### How the corpus is queried

`loghub` is both the corpus source and a Postgres (+pgvector) server: on
startup it fetches the pinned LogHub files as before, then loads every line
into a `lines(id, dataset, line_number, text)` table (`fetch_corpus.py`'s
`load_into_postgres()`). `datasetgen`'s easy tier (`question_generators.py`'s
`_count_matches`) and `schema_validator.py`'s evidence/numeric_claims checks
query that table over `datasetgen-net` via `pg_client.py` -- real
`SELECT ... WHERE strpos(...)`/`~*` SQL, not a file scan. Medium/hard
generation is unaffected: it still reads the raw `*_2k.log` files directly
(mounted read-only) for anchor matching and evidence windows.

### Manual SQL spot-check (optional)

`output/pilot/questions.json`'s answers can be double-checked by hand, from
the same `lines` table `pg_client.py` queries -- there's no separate
verification database (there used to be a second `sql_verify` Postgres
instance for this; it was redundant once loghub started serving the corpus
from Postgres itself, so it was removed).

```bash
# Run the 20 official-output checks (ILIKE queries matching
# question_generators.py's matching logic exactly; user/db match .env):
docker compose exec -T loghub psql -U loghub -d loghub < sql_verification/verify.sql

# Or connect interactively:
docker compose exec -it loghub psql -U loghub -d loghub
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
