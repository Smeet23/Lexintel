# Held-out retrieval benchmark

Thirty questions cover six new documents. The set contains ten judgment questions, ten contract questions, and ten pleading questions. All questions and gold source anchors were frozen before scoring. The [manifest](heldout.json) pins the originals by SHA-256 and records the exact passage or table row.

The corpus contains [Ram Kumar](https://api.sci.gov.in/supremecourt/2024/25762/25762_2024_5_1502_59288_Judgement_10-Feb-2025.pdf), [Chellammal](https://api.sci.gov.in/supremecourt/2019/45661/45661_2019_14_1501_61058_Judgement_22-Apr-2025.pdf), and four synthetic fixtures. The complaint and answer deliberately give conflicting accounts of the same facts. All three systems searched the same 136 chunks with the same source-offset scoring rule.

| Method                      | Evidence in top 20 | Evidence in top 5 | Evidence at rank 1 | Warm p50 | Warm p95 | Valid source references |
| --------------------------- | -----------------: | ----------------: | -----------------: | -------: | -------: | ----------------------: |
| BM25                        |              27/30 |             20/30 |               8/30 |   6.5 ms |  20.6 ms |                 148/148 |
| BM25 + MiniLM reranker      |              27/30 |             25/30 |              15/30 | 240.1 ms | 355.6 ms |                 148/148 |
| BM25 + E5 + MiniLM reranker |              30/30 |             26/30 |              16/30 | 250.6 ms | 351.2 ms |                 150/150 |

Warm latency sums local PostgreSQL search, query embedding and fusion when applicable, and reranking for one question. It excludes model loading, passage pre-indexing, network hops in the app, and answer generation. The experiment used the pinned English E5-small-v2 and MiniLM weights from the development benchmark. No OpenRouter requests were made.

Hybrid brought all three BM25 candidate misses into the top 20. Tideline’s requested replacement-purchase costs rose to rank 1. Riverside’s evidence-preservation request and the final order in Ram Kumar still fell outside the top five. The threshold-test precedent in Ram Kumar and the surrender exemption in Chellammal both landed sixth. This version of Ask AI retrieves and cites passages. An evaluation of generated answers and abstention remains separate.

## Per-question evidence ranks

A dash means the labeled evidence was absent from that setup’s top 20. Bold ranks fall in the first five.

| ID   | Gold source                                              |  BM25 | BM25 + reranker | Hybrid + reranker |
| ---- | -------------------------------------------------------- | ----: | --------------: | ----------------: |
| R1   | ram_kumar, ¶30, p. 21                                    |     – |               – |                18 |
| R2   | ram_kumar, ¶10, p. 7                                     |    12 |           **5** |             **5** |
| R3   | ram_kumar, ¶13, p. 8                                     |    14 |           **5** |             **5** |
| R4   | ram_kumar, ¶14, p. 8                                     |    13 |               6 |                 6 |
| R5   | ram_kumar, ¶11, p. 7                                     | **4** |           **4** |             **4** |
| C1   | chellammal, ¶31, p. 14                                   | **3** |           **5** |             **4** |
| C2   | chellammal, ¶32, p. 14                                   |     7 |               6 |                 6 |
| C3   | chellammal, ¶19, p. 8                                    | **1** |           **1** |             **1** |
| C4   | chellammal, ¶3, p. 2                                     | **2** |           **1** |             **1** |
| C5   | chellammal, ¶6, p. 2                                     | **1** |           **1** |             **1** |
| S1   | cold_storage, Schedule B, After-hours dispatch row, p. 1 | **2** |           **1** |             **1** |
| S2   | cold_storage, Schedule B, Missed collection row, p. 1    |     6 |           **4** |             **4** |
| S3   | cold_storage, §2.1, p. 1                                 | **1** |           **1** |             **1** |
| S4   | cold_storage, §2.2, p. 1                                 | **1** |           **1** |             **1** |
| S5   | cold_storage, §3.1, p. 1                                 | **1** |           **1** |             **1** |
| D1   | data_processing, §2.1, line 10                           | **4** |           **2** |             **2** |
| D2   | data_processing, §1.2, line 7                            | **3** |           **2** |             **2** |
| D3   | data_processing, §3.2, line 16                           | **2** |           **2** |             **2** |
| D4   | data_processing, §4.2, line 20                           | **2** |           **1** |             **1** |
| D5   | data_processing, §4.3, line 21                           | **4** |           **1** |             **1** |
| P1-H | riverside, ¶3, line 16                                   | **2** |           **1** |             **1** |
| P2-H | riverside, ¶6, line 19                                   | **4** |           **1** |             **1** |
| P3-H | riverside, ¶10, line 27                                  | **1** |           **1** |             **1** |
| P4-H | riverside, ¶9, line 24                                   |    17 |           **2** |             **2** |
| P5-H | riverside, ¶11, line 28                                  |     – |               – |                13 |
| T1   | tideline, ¶3, line 14                                    | **1** |           **1** |             **1** |
| T2   | tideline, ¶4, line 15                                    | **1** |           **1** |             **1** |
| T3   | tideline, ¶5, line 16                                    |     7 |           **1** |             **1** |
| T4   | tideline, ¶7, line 20                                    | **2** |           **2** |             **2** |
| T5   | tideline, ¶8, line 21                                    |     – |               – |             **1** |

## Reproduce

Ingest the six documents in the manifest and use a separate PostgreSQL database as in the [development benchmark](README.md). Pass the pinned local model directories from that benchmark.

```bash
PYTHONPATH=apps/api/src apps/api/.venv/bin/python apps/api/benchmarks/retrieval.py \
  --api http://127.0.0.1:8000 --dsn "$LEXINTEL_BENCH_DSN" \
  --suite apps/api/benchmarks/heldout.json --output /tmp/lexintel-heldout-results.json \
  --e5 "$LEXINTEL_E5_SNAPSHOT" --reranker "$LEXINTEL_RERANKER_SNAPSHOT"
```
