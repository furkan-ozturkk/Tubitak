# LogRouter Benchmark Extension

This repository is a project skeleton built around the paper **"LogRouter:
Adaptive Two-Level LLM Routing for Log Question Answering in Big Data
Systems"** (Coskuner, Zeybel, Dolan; TUBITAK BILGEM). The goal of this project
is to extend the paper's benchmark and dataset.

Most modules currently contain only a module-level docstring describing
their intended responsibility. `config.py`, `generation/ollama_client.py`,
`storage/loghub_loader.py`, `storage/structured_store.py`, and `main.py`
(`ingest` command) already have a working implementation; the rest are
still unimplemented.

## Relation to the paper

The paper evaluates a two-level cost-aware LLM router on four LogHub datasets
(Linux, Apache, Windows, Mac; 70 questions in total), reaching 88.4% mean
routing accuracy and an 18.6 second mean end-to-end latency (Table II). The
authors state that the implementation, evaluation harness, and the
LogHub-derived question set with gold routing labels will be released after
acceptance.

This project extends that scope in two directions: adding the remaining
Loghub-2.0 systems (HDFS, BGL, Hadoop, Spark, Zookeeper, HPC, Thunderbird,
OpenSSH, HealthApp, Proxifier, OpenStack) to the benchmark, and growing the
question set beyond the paper's original 70 questions.

## Architecture mapping

| Paper component | Used in the paper | Counterpart in this project |
|---|---|---|
| Log source | Grafana Loki (live stream) | `storage/loghub_loader.py` (static Loghub-2.0 datasets) |
| Keyword and SQL backend | Apache Druid | `storage/structured_store.py` (Postgres at this scale) |
| SQL generation | Coder LLM via Ollama | `generation/sql_generator.py` |
| Semantic answer generation | Qwen2.5-14B / Qwen3-32B via Ollama | `generation/semantic_generator.py` |
| Model serving | Ollama (embedding, 14B, 32B, coder) | `generation/ollama_client.py` |
| Question set (70 questions, 4 systems) | LogHub with gold routing labels | `benchmark/question_generator.py` |
| Task taxonomy | Seven task families | `benchmark/task_taxonomy.py` |
| Baselines | Fixed-14B, Fixed-32B | `evaluation/baselines.py` |
| Ablation study | Nine ablation conditions | `evaluation/ablation.py` |
| Result tables | Table II-V format | `evaluation/report_generator.py` |

## Current structure

```
logrouter-benchmark/
├── main.py
├── config.py
├── docker-compose.yml
├── requirements.txt
├── storage/
│   ├── loghub_loader.py
│   └── structured_store.py
├── generation/
│   ├── ollama_client.py
│   ├── sql_generator.py
│   └── semantic_generator.py
├── benchmark/
│   ├── question_generator.py
│   └── task_taxonomy.py
├── evaluation/
│   ├── baselines.py
│   ├── ablation.py
│   └── report_generator.py
├── docs/
│   ├── ollama_models.md
│   └── gpu_parallelism.md
└── utils/
    └── logging_utils.py
```

## Running the ingestion pipeline

With the containers from `docker-compose.yml` running:

```
docker exec -it logrouter-app python main.py ingest --dataset HDFS
```

This downloads the Loghub-2.0 structured CSV for the given system (or a
comma-separated list, for example `HDFS,BGL`) and writes it into the
`logs` table in Postgres, creating the schema on first run. See
`storage/loghub_loader.py` and `storage/structured_store.py` for the
implementation.

## Ollama connection

By default, `config.py` points `OLLAMA_HOST` at the team's shared GPU
machine (`http://10.15.33.66:11435`), reachable only while connected to the
team VPN. This can be verified with:

```
curl -s http://10.15.33.66:11435/api/tags
```

Set the `OLLAMA_HOST` environment variable to override this, for example to
point at a local Ollama instance (`http://localhost:11434`) when the VPN is
not available. Use a lighter model (`OLLAMA_LIGHT_MODEL`, default
`qwen2.5-coder:7b`) for experiments, and the full-size model
(`OLLAMA_DEFAULT_MODEL`, default `qwen2.5-coder:14b`) for the real
application, so the shared GPU is used efficiently. See
`docs/ollama_models.md` for the full model research and
`docs/gpu_parallelism.md` for how Ollama behaves on multi-GPU hardware.

## Planned modules

The following components appear in the paper's architecture and are planned
for a later iteration: Drain3 parsing, chunking and embedding, a vector
store (pgvector), hybrid retrieval, a two-level router (keyword signal
vocabulary and complexity-based model selection), full evaluation metrics
(ROUGE-1, BERTScore, RAGAS, LLM-judge Answer Correctness, and retrieval
metrics), and a test suite.

## Next steps

1. Implement the remaining generation modules (`sql_generator.py`,
   `semantic_generator.py`) against the Ollama client.
2. Collect the paper's four-dataset question set and extend it with new
   questions and gold routing labels for the remaining Loghub-2.0 systems.
3. Implement the evaluation metrics and reporting in the format of the
   paper's Table II-V.
4. Implement the two-level router (keyword signal vocabulary and
   complexity-based model selection).
