# LogRouter Benchmark Extension

This repository is a project skeleton built around the paper **"LogRouter:
Adaptive Two-Level LLM Routing for Log Question Answering in Big Data
Systems"** (Coskuner, Zeybel, Dolan; TUBITAK BILGEM). The goal of this project
is to extend the paper's benchmark and dataset.

All modules currently contain only a module-level docstring describing their
intended responsibility; no implementation has been added yet.

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
└── utils/
    └── logging_utils.py
```

## Planned modules

The following components appear in the paper's architecture and are planned
for a later iteration: ingestion (Drain3 parsing, chunking, and embedding),
a vector store (pgvector), hybrid retrieval, a two-level router (keyword
signal vocabulary and complexity-based model selection), full evaluation
metrics (ROUGE-1, BERTScore, RAGAS, LLM-judge Answer Correctness, and
retrieval metrics), and a test suite.

## Next steps

1. Implement `storage/structured_store.py` on top of the existing Loghub-2.0
   ingestion pipeline.
2. Implement the Ollama client and generation modules against a running
   Ollama instance.
3. Collect the paper's four-dataset question set and extend it with new
   questions and gold routing labels for the remaining Loghub-2.0 systems.
4. Implement the evaluation metrics and reporting in the format of the
   paper's Table II-V.
