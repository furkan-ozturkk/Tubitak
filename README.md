# LogRouter Evaluation Dataset — Production Architecture



## Layout

```text
.
├── main.py                    # the only executable entry point; contains only main()
├── generate.py                # generation library (Section 7)
├── validate.py                # validation library (Sections 2/6)
├── requirements.txt
├── config/
│   ├── args.py                # every CLI parameter, declared exactly once
│   └── question_schema.json   # the Section 6 record contract
├── src/
│   ├── params/                # one dataclass per config concern + get_*_params(args)
│   │   ├── corpus_params.py       generation_params.py    ollama_params.py
│   │   ├── results_params.py      review_params.py        scale_params.py
│   │   └── validation_params.py
│   ├── data/                  # the corpus itself
│   │   ├── corpus_loader.py   # bytes, lines, hashing
│   │   ├── data_factory.py    # corpus_provider(): the one place a log file is opened
│   │   └── dataset_specs.py   # curated per-dataset literals and grouping rules
│   ├── generators/            # the three question tiers
│   │   ├── easy_tier.py       medium_tier.py          hard_tier.py
│   └── utils/                 # helpers and remote-service clients
│       ├── helper_postgres.py # client: loghub's lines table
│       ├── helper_ollama.py   # client: the remote Ollama server
│       ├── helper_evidence.py helper_records.py       helper_review.py
│       ├── helper_run.py      # atomic writes, summaries
│       └── helper_splits.py   # leak-proof dev/test assignment
├── analysis/
│   └── analysis_tables.py     # offline composition tables; reads a finished dataset
├── scripts/
│   ├── run_all.sh             # generate -> validate -> tables
│   ├── check_smoke.sh         # imports, CLI, validations, comment policy
│   ├── check_status.sh        # what is in output/ right now
│   ├── check_no_comments.py   # enforces the docstring-only policy
│   ├── verify_answers.sql     # manual SQL spot-check of the official 20
│   └── tasks/                 # one file per pipeline stage
├── docker/
│   ├── compose.yml            # reproducible: runs the built image
│   ├── compose.dev.yml        # override: runs the working tree
│   ├── datasetgen.Dockerfile  loghub.Dockerfile      loghub-entrypoint.sh
│   └── .env.example
├── loghub/
│   ├── corpus_manifest.json   # pinned commit + 10 datasets
│   └── fetch_corpus.py        # fetch, checksum, lock verification, load into Postgres
├── tests/
└── output/
    ├── pilot/                 # the live dataset and its report
    └── archive/               # superseded first-pass artifacts (see its README)
```

### Conventions

**One entry point.** `main.py` holds `main()` and nothing else. `generate.py` and
`validate.py` are libraries; each command's work lives in the module that owns it.

**Parameters are dataclasses.** `config/args.py` declares every flag exactly once, and a
flag reaches code only through a `src.params` dataclass built by `get_*_params(args)`.
Defaults live on the dataclass, and an `argparse` default of `None` marks a field where
the dataclass default wins. There is no YAML: `config/question_schema.json` is the one
non-Python config artifact, and it exists only because `jsonschema` must be handed a
schema document.

**Python explains itself in docstrings, not comments.** Google-style, on the module,
class and function. `scripts/check_no_comments.py` enforces it and runs in the smoke
test. Shell, YAML, Dockerfiles and SQL keep comments, having nowhere else to put prose.

## Running it

```bash
cp docker/.env.example docker/.env    # set a real POSTGRES_PASSWORD
docker compose -f docker/compose.yml up --build
```

`loghub` fetches the corpus, starts Postgres (+pgvector), loads the corpus into it,
reports healthy, and keeps running. `datasetgen` starts once loghub is healthy and stays
up. It runs the code baked into its image; add `-f docker/compose.dev.yml` to run the
working tree instead.

Every operation is one `--command` of `main.py`. Defaults match the container paths, so
most need no flags — `--help` lists every override.

```bash
docker compose -f docker/compose.yml exec datasetgen python3 main.py --command check-ollama
docker compose -f docker/compose.yml exec datasetgen python3 main.py --command generate
docker compose -f docker/compose.yml exec datasetgen python3 main.py --command validate
docker compose -f docker/compose.yml exec datasetgen python3 main.py --command review-export
docker compose -f docker/compose.yml exec datasetgen python3 main.py --command review-apply --reviewer <name>
```

`--command generate --full` runs all three tiers and needs Ollama. It writes
`/output/pilot/questions_full.json`, not the official output: its medium and hard records
are model drafts at `review_status=in_review`, and landing them on top of 20 verified
records would replace a verified dataset with drafts. Passing `--dataset` explicitly still
overwrites, so overwriting has to be asked for.

One-shot corpus checks, needing neither Postgres nor datasetgen:

```bash
docker compose -f docker/compose.yml run --rm --no-deps loghub verify
docker compose -f docker/compose.yml run --rm --no-deps loghub fetch
```

Locally, without Docker:

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m unittest discover -s tests -t .
bash scripts/check_smoke.sh
bash scripts/check_status.sh
./.venv/bin/python analysis/analysis_tables.py
```

### How the corpus is queried

`loghub` is both the corpus source and a Postgres (+pgvector) server: on startup it
fetches the pinned files, then loads every line into
`lines(id, dataset, line_number, text)` with `UNIQUE (dataset, line_number)`. The easy
tier and `validate.py` query that table over `datasetgen-net` via
`src/utils/helper_postgres.py` — real `SELECT ... WHERE strpos(...)` SQL, not a file scan.
Medium and hard generation reads the raw `*_2k.log` files directly, because it needs
contiguous evidence windows and regex-grouped events rather than aggregates.

Generation, validation and the manual SQL spot-check all read the same table, so agreeing
with each other proves less than it appears to. `validate.py` therefore compares the
loaded table against the raw file it came from — row count and ordered content — before
believing any answer computed from it.

### Manual SQL spot-check (optional)

```bash
docker compose -f docker/compose.yml exec -T loghub psql -U loghub -d loghub < scripts/verify_answers.sql
```

## What validation actually establishes

`validate.py` runs five layers, in increasing order of what each can prove:

1. **Schema** — every record matches `config/question_schema.json` exactly
   (`additionalProperties: false` throughout). A `FormatChecker` is attached, so
   `format: "date-time"` is enforced rather than parsed and ignored.
2. **Cross-record** — id uniqueness; a hard question spans ≥2 evidence groups; a
   deterministic intent is reachable through ≥3 phrasing families (`--strict`);
   `review_status` is consistent with how the gold was produced.
3. **Split** — every stored split is recomputed from the whole record set and compared.
4. **Corpus integrity** — the loaded table equals the raw file.
5. **Answer** — every numeric claim is recomputed by SQL *and compared to
   `expected_answer`*; a `line_lookup` answer must equal a line the record cites.

Layer 5 is why the easy tier ships `verified` without a human: nothing is asserted that a
reader cannot reproduce. Layers 4 and 5 need the corpus and the database; pointing
`--corpus_dir` somewhere without `*.log` files is an error, not a schema-only pass.

## Scientific Integrity Rules

1. **Data source.** LogHub data is fetched only via `fetch_corpus.py`, from a pinned
   commit (Section 4.1). `--verify-only` never writes the lock and fails without one,
   and a file already on disk is accepted only when the lock already covers it —
   otherwise a first run downloads. `--trust-existing` allows the offline case and says
   so in the lock's `locked_from` field.
2. **Generator ≠ reviewer.** The drafting model and the groundedness-check model must be
   different families (Section 5.5/6). Enforced three ways: two separate methods reading
   two separate config fields, a name check in `config/args.py`, and a *family and
   digest* comparison against the server's own metadata in `helper_ollama` — two
   different tags can share a family, so comparing names alone is not enough.
3. **Access boundary.** LogRouter's source code, infrastructure and internal
   architecture are never used or referenced (Section 1.1/3.2).

## Known limitations

Recorded rather than hidden. Each is a real gap, not a rough edge.

**No publish step.** `review-apply` edits review statuses in place; it does not merge
verified records from a full pass into the official output. The staged-scaling story of
Section 3 ("reviewed medium/hard records fold into the official output, growing the 20
upward") therefore has no command behind it yet. What exists is: a full pass writes its
own file, review decisions are applied to that file with an audit trail, and moving
records into `output/pilot/questions.json` is manual.

**No quota planner.** `--target_total_questions` and the difficulty mix are reported next
to the realised counts and never enforced; the tiers produce what the curated specs and
the corpus allow. Hitting a ratio by discarding valid questions would be worse than
reporting the gap, but it does mean a 100-question target and a 70/20/10 mix are targets,
not guarantees. The first full pass produced 116 at roughly 78/17/5.

**The official 20 fails `--strict` by design.** It ships one phrasing per intent, and
Section 2/7.4 wants three. The validator states this in the finding text. A scaled run
has no such excuse.

**Hard-tier evidence is thinner than the questions imply.** Each group cites its first
few matching lines while the questions ask about volume, timing and an observation
window. The prompt now states each group's total line count and first/last line so the
comparison rests on recorded numbers, but the model still sees only a sample. Groundedness
results on the first pass reflect it (OpenSSH 0/3 fully supported, BGL 1/5). These records
stay `in_review` for that reason.

**Claim-to-evidence mapping is coarse.** The groundedness check returns a verdict over the
whole evidence block and does not identify which line supports a claim, so a claim records
the group ids it could have drawn on (`candidate_group_ids`) rather than a specific line.
Naming one line would assert a precision the check did not establish.

**No integration tests.** `tests/` covers the pure logic with a fake corpus repository.
The corpus-to-database equality check cannot be meaningfully tested that way — faking the
database is the exact failure the check exists to catch — so it needs a suite against a
real loaded Postgres, which does not exist yet.

**`created_at` is a fixed future-dated constant.** `2026-08-01T00:00:00Z`, held constant so
repeated runs are byte-identical (Section 6). It is not when a record was generated. The
real timestamp of a *validation* is not recorded either; what a report does carry is the
code version and the digests of every input it read.

**Base images are pinned by tag, not digest.** `python:3.11-slim` and
`pgvector/pgvector:pg16`. Python dependencies are pinned exactly; there is no transitive
lock file and no SBOM.

**Ollama is plain HTTP with no authentication.** The default endpoint is a fixed private
address. Responses can be modified in transit, and the URL is logged, so credentials
embedded in it would leak. The server is outside this project's control (Section 5.5).

**pgvector is enabled but unused.** No vector column and no vector query exists yet; the
extension is in place for later semantic retrieval work.
