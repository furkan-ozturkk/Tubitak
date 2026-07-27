# Archived first-pass artifacts

Superseded outputs, kept because they are the project's record of what was produced
before the current pipeline existed. Nothing here is regenerable: the modules that
wrote these files (`layer1_deterministic.py`, `layer2_semantic.py`, `layer3_hard.py`)
were merged away, so their `gold_provenance.created_by` values point at paths that no
longer exist. That is accurate provenance, not a stale string to fix — the files really
were written by that code.

The live output is `output/pilot/`. Nothing reads this directory.

## `v1_116/` — the first full three-tier pass, 116 questions

| File | Records | What it is |
|---|---|---|
| `questions.json` | 116 | The merged pass. Was `output/pilot/questions_full_v1_116.json`. |
| `layer1.json` | 90 | Easy tier, as written by `layer1_deterministic.py`. |
| `layer2.json` | 20 | Medium tier, as written by `layer2_semantic.py`. |
| `layer3.json` | 3 | Hard tier — **incomplete, see below**. |
| `layer1_validation.json` | — | Validation report for `layer1.json`. |
| `layer2_validation.json` | — | Validation report for `layer2.json`. |
| `sample_ids.json` | 28 | Ids of the advisor-review sample. |
| `sample_for_review.md` | 28 | That sample, rendered for reading. |

### Known defects in this pass

These are recorded rather than repaired. Repairing them would mean rewriting files whose
value is being the artifact that was actually produced.

**`layer3.json` holds 3 of the 6 hard questions.** The merged `questions.json` contains
six; the per-tier file contains three. Missing from it:
`hadoop_v1_hard_container_task_compare`, `hdfs_v1_hard_block_lifecycle_compare`,
`zookeeper_v1_hard_ensemble_member_compare`. The three layer files sum to 113, not 116.
Treat `questions.json` as authoritative for this pass and the layer files as partial
views of it.

**No validation report covers the merged pass.** `layer1_validation.json` and
`layer2_validation.json` each cover one tier; nothing covers `questions.json` or the hard
tier. Neither report records the digest of the file it validated, so neither can prove
which bytes it certified. Reports written by the current `validate.py` carry
`config_snapshot.input_digests` for exactly that reason.

**One record's split leaks its event set.**
`hadoop_v1_hard_container_task_compare` is stored as `split: test`, but it cites two
container groups that other records also cite, and that connected event set resolves to
`dev`. The old generator assigned a hard question's split from its *first* group id
alone, so co-cited groups could land on opposite sides of the boundary — the same event
reachable from both dev and test. `src/utils/helper_splits.py` now assigns one split per
connected component of co-cited groups, and `validate.py` recomputes and compares it, so
this record fails validation today. Running the current validator over this file is
expected to report it.

**Medium and hard records are `review_status=in_review`.** They were never taken through
human review, and their `gold_provenance.method` is `independent_model_then_human` while
`reviewers` names only the generating script. The current validator rejects that
combination once a record claims `verified`; these records do not claim it, so they are
honest — just unfinished.

## `mini20/` — 20-question advisor sample, 20 records

| File | What it is |
|---|---|
| `questions.json` | 14 easy + 4 medium + 2 hard, drawn from `v1_116/questions.json`. |
| `validation_report.json` | Its validation report. |
| `mini20_for_review.md` | The sample, rendered for reading. |

Drawn to cover all ten LogHub datasets at roughly the configured 70/20/10 mix. Unrelated
to the official stage-1 output in `output/pilot/`, which is a different set built by a
different path: 20 records, all easy, all deterministic, all `verified`.
