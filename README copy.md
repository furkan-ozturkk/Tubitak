# LogRouter Benchmark Extension – Proje Mimarisi (Taslak)

Bu klasor, **"LogRouter: Adaptive Two-Level LLM Routing for Log Question Answering
in Big Data Systems"** (Coskuner, Zeybel, Dolan - TUBITAK BILGEM) makalesindeki
sistemin mimarisini referans alarak, makaledeki **benchmark ve veri kumesini
genisletme** hedefi icin hazirlanan proje iskeletidir.

Tum dosyalar su an **bos / sadece aciklama docstring'i iceren** taslaklardir -
amac, hocaya "proje mimarisi bu sekilde olacak" diye gosterebilmek. Hicbir modulde
henuz gercek implementasyon yok.

## Makale ile iliski

Makale, 4 LogHub veri seti (Linux, Apache, Windows, Mac - toplam 70 soru) uzerinde
iki seviyeli bir router (L1: GENERAL/KEYWORD/SQL/SEMANTIC, L2: 14B/32B model secimi)
ile %88.4 ortalama dogruluk ve 18.6 sn uctan uca gecikme elde ediyor (bkz. Tablo II).
Acik bilim (Open Science) bolumunde yazarlar, uygulamayi, degerlendirme setini ve
altin (gold) routing etiketli LogHub soru setini kabul sonrasi acik kaynak olarak
yayinlayacaklarini belirtiyor.

**Bu projenin hedefi:** ayni mimariyi (veya kucuk olcekli bir yeniden uretimini)
kurup, makaledeki 4 veri seti / 70 soruluk kapsami, Loghub-2.0'daki diger sistemlere
(HDFS, BGL, Hadoop, Spark, Zookeeper, HPC, Thunderbird, OpenSSH, HealthApp,
Proxifier, OpenStack) ve daha fazla soruya genisletmek.

## Mimari esleme (makale -> bu proje)

| Makaledeki bilesen | Makalede kullanilan | Bu projede karsiligi |
|---|---|---|
| Log kaynagi | Grafana Loki (canli akis) | `ingestion/source_loader.py` - Loghub-2.0 statik verisetleri (bkz. onceki `dataset-pipeline`) |
| Normalizasyon | PySpark ingester | `ingestion/normalizer.py` |
| Structured branch | Drain3 (PySpark stage) | `ingestion/drain3_parser.py` |
| Semantic branch | Ollama nomic-embed-text + chunker | `ingestion/chunker_embedder.py` |
| Keyword/SQL backend | Apache Druid | `storage/structured_store.py` (kucuk olcekte: Postgres) |
| Semantic index | PostgreSQL + pgvector | `storage/vector_store.py` |
| Hybrid retrieval | pgvector + FTS + RRF (k=60) | `retrieval/hybrid_retrieval.py` |
| Level-1 router | Regex tabanli P0-P7 sozlugu | `router/keyword_patterns.py`, `router/level1_router.py` |
| Level-2 router | Karmasiklik skoru -> 14B/32B | `router/level2_router.py` |
| Ureticiler | Qwen2.5-14B, Qwen3-32B, Qwen2.5-Coder-14B (Ollama) | `generation/ollama_client.py`, `sql_generator.py`, `semantic_generator.py` |
| Soru seti (70 soru, 4 sistem) | LogHub + altin routing etiketi | `benchmark/question_sets/`, `benchmark/datasets/` |
| Degerlendirme | Routing acc., ROUGE-1, BERTScore, RAGAS, Answer Correctness, Hit@k/Recall@k/MRR, gecikme | `evaluation/metrics/*` |
| Baseline'lar / ablation | Fixed-14B, Fixed-32B, 9 kosullu ablation | `evaluation/baselines.py`, `evaluation/ablation.py` |

## Klasor yapisi

```
logrouter-benchmark/
├── main.py                    # TEK giris noktasi (ingest / index / route / evaluate / ablate)
├── config.py                  # merkezi ayarlar
├── docker-compose.yml         # ollama + postgres(pgvector) altyapisi
├── ingestion/                 # Sekil 1 - Indexing Pipeline
│   ├── source_loader.py
│   ├── normalizer.py
│   ├── drain3_parser.py       # structured branch
│   └── chunker_embedder.py    # semantic branch
├── storage/
│   ├── structured_store.py    # keyword + SQL backend
│   └── vector_store.py        # pgvector semantic index
├── retrieval/
│   └── hybrid_retrieval.py    # dense + FTS + RRF
├── router/                    # Sekil 2 - Query-time Routing
│   ├── keyword_patterns.py    # P0-P7 (Tablo I)
│   ├── level1_router.py       # GENERAL/KEYWORD/SQL/SEMANTIC
│   └── level2_router.py       # 14B/32B secimi
├── generation/
│   ├── ollama_client.py
│   ├── sql_generator.py
│   └── semantic_generator.py
├── benchmark/                 # veri kumesi/soru seti genisletme
│   ├── datasets/
│   │   ├── linux/ apache/ windows/ mac/   # makaledeki mevcut 4 set
│   │   └── extended/                       # yeni eklenecek Loghub-2.0 sistemleri
│   ├── question_sets/
│   ├── question_generator.py
│   └── task_taxonomy.py       # 7 gorev ailesi
├── evaluation/                # Tablo II-V
│   ├── metrics/                # routing / lexical / semantic / ragas / judge / retrieval / latency
│   ├── baselines.py            # Fixed-14B / Fixed-32B
│   ├── ablation.py             # 9 kosullu ablation
│   └── report_generator.py
├── models/                     # Question, RouteResult veri siniflari
├── utils/
└── tests/
```

## Siradaki adimlar (bu iskelet onaylandiktan sonra)

1. `ingestion/` + `storage/`: mevcut `dataset-pipeline` (Loghub-2.0 -> Postgres) mantiginin
   buraya tasinmasi, `pgvector` uzantisinin Postgres'e eklenmesi, Drain3 entegrasyonu.
2. `router/`: Tablo I'deki P0-P7 regex sozlugunun ve Level-2 karmasiklik skorunun
   birebir uygulanmasi.
3. `benchmark/`: makaledeki 4 sistemin (Linux/Apache/Windows/Mac) soru setinin
   toplanmasi + Loghub-2.0'daki diger sistemler icin yeni soru/gold-label uretimi.
4. `evaluation/`: ROUGE-1/BERTScore/RAGAS/LLM-judge metriklerinin ve ablation
   kosullarinin uygulanmasi, makaledeki Tablo II-V formatinda raporlama.
