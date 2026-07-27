# LogRouter Evaluation Dataset

Produces the black-box question-answer dataset that LogRouter is evaluated against.

Ten LogHub log datasets are fetched from a pinned commit and loaded into Postgres.
Questions are then generated over them in three difficulty tiers, each answerable by
a different retrieval path — `sql` for aggregates, `keyword` for line lookup,
`semantic` for explanation and synthesis — so a router can be scored on whether it
picks the right one and returns the right answer.

Every answer is either machine-derived and re-checkable, or model-drafted and marked
as awaiting a human. Nothing in between. The project has no connection to
LogRouter's source code or infrastructure.

## Layout

```text
.
├── main.py                    # only entry point: reads args into every config dataclass, dispatches
├── generate.py                # question generation, three tiers
├── validate.py                # schema, cross-record, split, corpus and answer checks
├── verify_answers.py          # independent check: a model writes the SQL, result vs gold
├── requirements.txt
├── config/
│   ├── args.py                # every CLI parameter, declared exactly once
│   └── question_schema.json   # the record contract
├── src/
│   ├── params/                # one dataclass per config concern + get_*_params(args)
│   ├── corpus/                # pinned manifest + the fetcher (runs in the loghub container)
│   ├── data/                  # corpus loading, hashing, per-dataset question specs
│   ├── generators/            # easy_tier.py, medium_tier.py, hard_tier.py
│   └── utils/                 # clients (postgres, ollama) and helpers
├── analysis/
│   └── analysis_tables.py     # offline composition tables over a finished dataset
├── scripts/
│   ├── run_all.sh             # generate -> validate -> tables
│   ├── check_smoke.sh         # imports, CLI, validations, unit tests
│   ├── check_status.sh        # what is in output/ right now
│   └── tasks/                 # one file per pipeline stage
├── docker/
│   ├── compose.yml            # runs the built image
│   ├── compose.dev.yml        # override: runs the working tree
│   ├── datasetgen.Dockerfile  loghub.Dockerfile      loghub-entrypoint.sh
│   └── .env.example
├── tests/
└── output/
    ├── pilot/                 # the live dataset and its reports
    └── archive/               # superseded first-pass artifacts
```

Two conventions worth knowing before reading the code. `main.py` holds only `main()`,
which reads the parsed arguments into one dataclass per concern and then dispatches;
a parameter reaches a library only through that dataclass. And Python explains itself
in Google-style docstrings rather than `#` comments, which `tests/test_no_comments.py`
enforces.

## Running it

```bash
cp docker/.env.example docker/.env    # set a real POSTGRES_PASSWORD
docker compose -f docker/compose.yml up --build
```

`loghub` fetches the pinned corpus, starts Postgres (+pgvector), loads every log line
into a `lines` table, and stays up. `datasetgen` starts once loghub is healthy. It runs
the code baked into its image; add `-f docker/compose.dev.yml` to run the working tree
instead.

Each operation is one `--command`. Defaults match the container paths, so most need no
flags; `--help` lists every override.

```bash
E="docker compose -f docker/compose.yml exec datasetgen python3 main.py"

$E --command check-ollama      # connectivity, required models, model families
$E --command generate          # write the dataset (20-question stage-1 set)
$E --command generate --full   # all three tiers, needs Ollama, writes its own file
$E --command validate          # schema + cross-record + split + corpus + answer checks
$E --command verify-answers    # a model writes the SQL; its result vs the gold answer
$E --command review-export     # in_review records out to a CSV worksheet
$E --command review-apply --reviewer <name>
```

Corpus checks that need neither Postgres nor datasetgen:

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

## What the pipeline guarantees

**The corpus cannot change silently.** It is fetched only from the commit pinned in
`src/corpus/corpus_manifest.json`, every file's digest is locked, and `verify` fails
rather than writing a lock it does not have.

**Deterministic answers are re-derived, not trusted.** `validate` recomputes every
count from SQL and compares it to the answer, checks a lookup answer against the line
it cites, and compares the loaded table against the raw file before believing either.
`verify-answers` then re-derives the same answers through a model that writes its own
SQL and never sees the gold, so agreement means two independent routes landed on the
same value.

**A model never certifies its own answer.** The drafting model and the checking model
must be different families, verified against the server's own metadata rather than by
comparing names.

**Model-drafted answers stay `in_review`** until a named human accepts them, and the
decision is logged with their identity, the time, and the digest of the draft they
judged.

**Linked events cannot straddle the dev/test split.** Questions whose evidence groups
are cited together are assigned one split as a set, and `validate` recomputes it.

Gaps are recorded in `output/archive/README.md` and in the module docstrings of the
code that owns them — most notably: there is no publish step folding reviewed records
into the official output, the difficulty mix is reported rather than enforced, and
there are no integration tests against a live Postgres.
