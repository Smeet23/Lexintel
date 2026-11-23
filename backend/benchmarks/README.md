# LexIntel retrieval benchmark

The benchmark covers 19 questions across two Indian Supreme Court judgments, a synthetic patent complaint, and a synthetic agreement. The [question set](questions.json) pins the four originals by SHA-256 and labels the passage, page, or table row for each question. Eighteen questions were written before the first run. K7 was added as a targeted diagnostic for a known split in paragraph 3.1.

| Retrieval | Evidence hit at 5 | Evidence hit at 1 | Valid source references | Duplicate source characters |
| --- | ---: | ---: | ---: | ---: |
| BM25, zero overlap | 15/19 | 8/19 | 95/95 | 218/49,397 (0.44%) |
| BM25, sentence overlap | 16/19 | 8/19 | 95/95 | 315/49,579 (0.64%) |
| Local E5-small-v2 | 15/19 | 8/19 | 95/95 | 285/56,051 (0.51%) |
| BM25 + E5, reciprocal rank fusion | 16/19 | 8/19 | 95/95 | 337/49,548 (0.68%) |

An evidence hit requires a passage from the correct document whose source spans contain every token of the labeled anchor on the gold page. Table questions also require the exact parsed row. Multi-part questions require the additional answer text in the same passage. A valid source reference means the span text matches its stored page offsets. Duplicate characters count repeated source coordinates within each question's five returned passages, including repeated table headers. All runs used the same corpus. The BM25 runs differed only in judgment overlap and both indexed 119 chunks.

K7 asks about two adjacent sentences in the quoted police complaint. With zero overlap, the best hit supplies the damage but omits the intervention. With overlap, rank 2 supplies both details in one passage. On the original 18 questions the two BM25 settings retrieved the same top five passages and tied at 15/18.

For K1, both rankers favored case background and missed the final order for a question phrased as pre-arrest protection. For B2, both missed the insurer’s proposal to average two returns. For B4, E5 found the High Court’s explanation at rank 3, while BM25 and rank fusion missed it. These misses point to role-aware ranking for orders and party submissions, and a controlled held-out test for wording differences.

At the time of this experiment, Ask AI used BM25 with sentence overlap for newly parsed judgments. Rank fusion tied on hits and returned a slightly larger share of duplicate source text. The app returns ranked passages with source links and makes no model calls. The English E5 experiment used public weights locally. This is a small development set with one targeted diagnostic, so the scores do not estimate performance on unseen matters.

## Question results

A dash means the gold evidence was outside the top five.

| ID | Gold locator | Zero overlap | Sentence overlap | E5 | Fusion |
| --- | --- | ---: | ---: | ---: | ---: |
| K1 | kiran, ¶10, p. 23 | – | – | – | – |
| K2 | kiran, ¶4.2, p. 8 | 2 | 2 | – | 3 |
| K3 | kiran, ¶5.6, p. 14 | 1 | 1 | 1 | 1 |
| K4 | kiran, ¶8, p. 21 | 4 | 4 | 2 | 2 |
| K5 | kiran, ¶10.1, p. 23 | 1 | 1 | 2 | 1 |
| K6 | kiran, ¶6.2, p. 19 | 2 | 2 | 3 | 2 |
| K7 | kiran, ¶3.1, p. 3, adjacent sentences | – | 2 | 1 | 1 |
| B1 | bhargava, ¶16, p. 11 | 3 | 3 | – | 4 |
| B2 | bhargava, ¶10, p. 6 | – | – | – | – |
| B3 | bhargava, ¶16, p. 11 | 2 | 2 | 4 | 2 |
| B4 | bhargava, ¶6, p. 4 | – | – | 3 | – |
| P1 | complaint, ¶22 | 1 | 1 | 3 | 2 |
| P2 | complaint, ¶17 | 2 | 2 | 1 | 2 |
| P3 | complaint, ¶33 | 1 | 1 | 1 | 1 |
| P4 | complaint, ¶30 | 1 | 1 | 1 | 1 |
| A1 | agreement, §7.2, p. 1 | 1 | 1 | 1 | 1 |
| A2 | agreement, Schedule A, Late delivery row, p. 1 | 1 | 1 | 1 | 1 |
| A3 | agreement, Schedule A, Early termination row, p. 1 | 2 | 2 | 2 | 2 |
| A4 | agreement, Effective Date, p. 1 | 1 | 1 | 1 | 1 |

## Candidate depth and reranking

The follow-up checked the top 20 candidates on the same 19 questions, using the sentence-overlap corpus for all comparisons below. The local MiniLM cross-encoder scored each question together with each candidate passage. Both reranked runs started from the top 20 of their respective retrieval lists.

| Retrieval | Evidence hit at 20 | Evidence hit at 5 | Evidence hit at 1 |
| --- | ---: | ---: | ---: |
| BM25 | 18/19 | 16/19 | 8/19 |
| BM25 + MiniLM reranker | 18/19 | 17/19 | 11/19 |
| BM25 + E5 fusion | 19/19 | 16/19 | 8/19 |
| BM25 + E5 fusion + MiniLM reranker | 19/19 | 18/19 | 11/19 |

| Original BM25 miss | BM25 rank | E5 rank | Hybrid rank | BM25 + reranker | Hybrid + reranker |
| --- | ---: | ---: | ---: | ---: | ---: |
| K1 · The final order cancelling pre-arrest bail | 13 | 14 | 8 | 7 | 7 |
| B2 · The insurer’s proposed method for estimating earnings | – | 11 | 17 | – | 5 |
| B4 · The High Court’s reason for using the older tax return | 10 | 3 | 7 | 4 | 4 |

A dash means the gold passage was outside the top 20. B4 was a ranking failure that BM25 reranking corrected. B2 required expanding the candidate pool, then reranking. K1 improved to rank 7 and remains a top-five miss. Every reranked result still passed the source-offset check. The app served BM25 during this development experiment. See the [held-out benchmark](HELDOUT.md) for the follow-up. The reranker and hybrid runs were offline experiments on the development set, which includes the targeted split-paragraph question.

## Reproduce

Upload the four originals linked in [questions.json](questions.json) to the running local app and wait for Parsed. The runner identifies the documents by their hashes and can read them across multiple matters. Use a separate PostgreSQL database with pgvector and pg_textsearch available. Both variants run in isolated schemas that are rolled back after scoring.

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/benchmarks/retrieval.py \
  --api http://127.0.0.1:8000 --dsn "$LEXINTEL_BENCH_DSN" \
  --output /tmp/lexintel-retrieval-results.json
```

To reproduce the optional E5 and fusion scores, install the existing parser dependency group and pass `--e5` with a local snapshot of [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2/tree/ffb93f3bd4047442299a41ebb6fa998a38507c52). The encoder loads files locally and uses the model’s query and passage prefixes. No paid provider is needed for any run.

Pass `--reranker` with a local snapshot of [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) to score the reranked BM25 run. Passing both model paths also scores reranked hybrid retrieval.
