# Filing-aware retrieval benchmark

The Google v. Oracle pilot uses the existing matter with 62 files and 4,313 indexed passages. Its [20-question manifest](oracle.json) pins the answer-bearing files by SHA-256 and records exact pages and text anchors. It covers pleadings, trial and appellate rulings, signed verdicts, jury instructions, a blank verdict form, merits briefs, and a dissent. A document can only earn credit for an alternative location when the manifest explicitly lists that passage or a verified copy of it. The verdict questions also require the answer box's literal marker to be present in the cited source span.

All four variants use the same BM25 index, local E5-small-v2 embeddings, and MiniLM cross-encoder. The runner opens a read-only database transaction and loads the model weights from the local cache. It makes zero OpenRouter requests.

1. **Baseline.** Retrieve the existing hybrid pool of up to 20 passages and apply the cross-encoder.
2. **Filing metadata.** Reorder the same pool using soft boosts for type, court, year, and filing party.
3. **Filing and passage labels.** Add boosts for functions such as a verdict, blank form, or disposition, and penalize certain mismatched roles such as a dissent for a majority holding question.
4. **Scoped candidates.** Keep the original candidates and append up to eight from filings matching the routed attributes. Rerank the union and apply both kinds of boosts.

The runner first checks whether each gold answer is present anywhere in the index. A hit is an exact evidence span in the right document and page. The expanded pool can hold 28 passages, so the full-pool count is reported alongside an equal-depth top-20 score. Citation validity checks page and character offsets against the stored page text.

The pilot treats descriptive filenames and page headers as classification inputs. A second matter with raw docket uploads and generic filenames is needed to test classification reliability. The Google v. Oracle questions were used to develop and debug the prototype, including verifying alternative passages and duplicate appendix sources. The [30-question regression suite](heldout.json) retains its original gold labels.

## Google v. Oracle results

| Variant                   | Rank 1 | Top 5 | Top 20 | In candidate pool |
| ------------------------- | -----: | ----: | -----: | ----------------: |
| Baseline                  |   5/20 |  7/20 |   9/20 |              9/20 |
| Filing metadata           |   8/20 |  8/20 |   9/20 |              9/20 |
| Filing and passage labels |   8/20 |  9/20 |   9/20 |              9/20 |
| Scoped candidates         |   9/20 | 14/20 |  15/20 |             16/20 |

All 20 answers were present in the index. All four variants returned 100 of 100 valid page and offset references. This measures whether the reference resolves to its stored source span. The exact evidence ranks measure whether that span actually answers the question.

- **Signed 2016 verdict.** The baseline pool missed it. Filing-scoped retrieval moved the checked verdict in ECF 1982 to rank 1. The blank form in ECF 1981 reached rank 3 for the separate question asking how the form was worded.
- **District court copyrightability order.** The exact rationale entered the expanded pool at rank 25. The Federal Circuit opinion appeared first, so the system still risked answering from a different court.
- **Filing identified, supporting passage absent.** For four questions, the expanded variant picked the right filing first but missed the labeled answer throughout the pool. The topics were the fourth fair-use factor, the 2018 appellate disposition, and two points from Oracle’s merits brief.

On this local run, warm p95 was 2.22 seconds for the baseline and 2.97 seconds for scoped candidates, measured through reranking. The scoped prototype is close to the application’s three-second ranking request timeout even before network overhead.

| ID  | Baseline | Filing metadata | Filing and passage labels | Scoped candidates |
| --- | -------: | --------------: | ------------------------: | ----------------: |
| O01 |        – |               – |                         – |                 1 |
| O02 |        – |               – |                         – |                 3 |
| O03 |        – |               – |                         – |                 – |
| O04 |       16 |               7 |                         3 |                 3 |
| O05 |        1 |               1 |                         1 |                 1 |
| O06 |        – |               – |                         – |                 2 |
| O07 |        7 |               1 |                         1 |                 1 |
| O08 |        – |               – |                         – |                 1 |
| O09 |        – |               – |                         – |                25 |
| O10 |        4 |               1 |                         1 |                 1 |
| O11 |        – |               – |                         – |                 – |
| O12 |        1 |               1 |                         1 |                 1 |
| O13 |        – |               – |                         – |                 3 |
| O14 |        1 |               1 |                         1 |                 1 |
| O15 |        1 |               1 |                         1 |                 1 |
| O16 |        – |               – |                         – |                 – |
| O17 |        – |               – |                         – |                11 |
| O18 |        – |               – |                         – |                 – |
| O19 |        5 |               1 |                         1 |                 3 |
| O20 |        1 |               1 |                         1 |                 1 |

A dash means the exact evidence was absent from the variant’s entire candidate pool.

## Regression on the earlier suite

| Variant                   | Rank 1 | Top 5 | Top 20 |
| ------------------------- | -----: | ----: | -----: |
| Baseline                  |  16/30 | 26/30 |  30/30 |
| Filing metadata           |  16/30 | 26/30 |  30/30 |
| Filing and passage labels |  17/30 | 26/30 |  30/30 |
| Scoped candidates         |  17/30 | 26/30 |  30/30 |

Chellammal C1 moved from rank 5 to rank 1. The passage containing the remand order received the disposition boost. The 16 backend tests also passed with a disposable PostgreSQL database.

## Reproduce

Start the local Compose services with both benchmark matters imported and their local vector indexes ready. Use the matter IDs from the local installation.

```bash
mkdir -p /tmp/lexintel-filing-benchmark
docker compose run --rm --no-deps -T \
  -v "$PWD/apps/api:/bench:ro" \
  -v /tmp/lexintel-filing-benchmark:/results \
  -e PYTHONPATH=/bench/src -e HF_HUB_OFFLINE=1 ranking \
  python /bench/benchmarks/filing_retrieval.py \
    --matter 525d5fcc-a847-44f4-a135-c91badf70dc3 \
    --suite /bench/benchmarks/oracle.json --output /results/oracle.json
```

Repeat with matter `053b86cb-6736-4c72-b07e-95325abaf5eb` and the suite `/bench/benchmarks/heldout.json` for the regression set. The JSON output includes the exact evidence rank, the candidate pool size, whether the answer was indexed, and the file, page, and label for each of the five results.
