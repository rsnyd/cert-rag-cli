
# Evaluation Report: cert-rag-cli (SQF certification corpus)

## Method

A 34-question golden set over our SQF certification documents, plus 5
out-of-corpus probe questions where the correct behavior is refusal. Scored
questions span easy (11, single clause lookup), medium (12, synthesis across 2-3
clauses), and hard (11, edge cases, comparisons, troubleshooting), tagged by
requirement area.

Two kinds of measurement:

1. Deterministic, no LLM call. `clause_hit@k` asks whether the clause that
   actually contains the requirement was retrieved in the top k. `refusal rate`
   asks whether the system declined on the probe set. Both are stable across
   runs and immune to judge drift.
2. LLM-as-judge. Claude Sonnet 4.6 scores five axes 1-5 with brief reasoning,
   given the question, the reference answer, and the excerpts the system
   actually retrieved: factual correctness, completeness, relevance, citation
   correctness, and grounding.

Citation correctness and grounding exist because in a compliance setting a
fluent answer citing the wrong clause is worse than a refusal. The three
standard axes do not catch that failure.

A third metric was added during this work: **answer cites expected clause**,
which parses clause numbers out of the answer text rather than reading chunk
metadata. `clause_hit@k` cannot compare configurations that chunk differently,
because it reads a metadata field one of them does not emit. This one is
chunk-size independent and is the metric that decided the chunking question.

## Corpus

44 SQF documents (40 DOCX, 4 PDF, 243 pages). None required OCR - all four PDFs
carry a native text layer at 1,400-3,500 chars/page, so the `ocrmypdf` path in
`ingest.py` exists but never fired on this corpus.

Chunked on clause boundaries rather than a fixed window, yielding 971 chunks
(mean 689 chars) indexed in Chroma with clause, page, module and doc_type
metadata. Clause attribution: 816 chunks from a header in the text, 87 from the
filename, 68 with no clause (front matter and unnumbered prose).

## Baseline configuration

- Chunking: clause-aware, sub-split above 2400 characters
- Embedding: Voyage voyage-3-lite
- Retrieval: cosine similarity, top_k=5
- Generation: Claude Sonnet 4.6, refusal-first system prompt

## Baseline results

Compliance metrics:

| Metric                      | Value |
| --------------------------- | ----- |
| Probe refusal rate          | 5/5   |
| False refusal rate (scored) | 1/34  |
| clause hit@1                | 85.3% |
| clause hit@3                | 97.1% |
| clause hit@5                | 97.1% |
| Answer cites expected clause| 97.1% |

Judged axes:

| Axis                 | Score | Notes                                                    |
| -------------------- | ----- | -------------------------------------------------------- |
| Citation correctness | 4.50  | Clause metadata survives retrieval into the prompt header |
| Grounding            | 4.79  | Highest axis; the refusal-first prompt is doing its job   |
| Factual correctness  | 4.50  |                                                          |
| Completeness         | 3.88  | Weakest axis by 0.6 - diagnosed below, not a model defect |
| Relevance            | 4.53  |                                                          |
| Overall              | 4.44  |                                                          |

By difficulty:

| Difficulty | Overall | hit@3 |
| ---------- | ------- | ----- |
| Easy       | 4.87    | 100%  |
| Medium     | 4.33    | 92%   |
| Hard       | 4.13    | 100%  |

Retrieval is near-ceiling at every difficulty; the overall gradient comes
entirely from generation. Hard questions retrieve the right clause 100% of the
time and still score lowest, so remaining headroom is in synthesis, not search.

Worst requirement areas by tag:

| Tag                 | Overall | Citation | Count |
| ------------------- | ------- | -------- | ----- |
| sanitation          | 3.40    | 3.00     | 1     |
| pest-control        | 3.40    | 3.00     | 1     |
| allergen-management | 3.60    | 3.00     | 1     |
| training            | 3.87    | 3.67     | 3     |
| food-safety-plan    | 4.07    | 4.00     | 6     |
| corrective-action   | 4.16    | 4.40     | 5     |
| verification        | 4.20    | 4.57     | 7     |

The first three are single-question tags and should not be read as areas of
weakness - one bad answer sets the mean. Only `verification` (n=7),
`food-safety-plan` (n=6) and `corrective-action` (n=5) have enough questions to
be worth acting on.

## Judge variance

Two runs of the identical configuration against the identical index, a day
apart, to establish the noise floor before interpreting any delta:

| Axis      | Run 1 | Run 2 | Drift |
| --------- | ----- | ----- | ----- |
| Citation  | 4.50  | 4.50  | 0.00  |
| Grounding | 4.79  | 4.85  | +0.06 |
| Factual   | 4.50  | 4.50  | 0.00  |
| Complete  | 3.88  | 3.85  | -0.03 |
| Relevant  | 4.53  | 4.50  | -0.03 |
| Overall   | 4.44  | 4.44  | 0.00  |

Maximum drift 0.06, so anything above ~0.1 on these axes is signal. This is what
makes the two experiments below interpretable at n=34.

## Experiment 1: fixed 2000-char sliding window (chunking ablation)

Clause boundaries and clause metadata both discarded; everything upstream of the
split held identical. 381 chunks (mean 1,887 chars) versus 971.

Delta vs baseline:

- clause hit@3: 97.1% -> 0% (definitional, see below)
- Probe refusal: 5/5 -> 5/5 (no change)
- Citation: -0.47
- Grounding: -0.14
- Overall: -0.09
- **Answer cites expected clause: 97.1% -> 73.5%** (9 wins vs 1, exact McNemar p=0.022)

Three observations:

1. **Two headline metrics were artifacts, not behavior.** `clause_hit@k` reads
   `chunk["clause"]`, which a fixed window does not emit, so 0% measures the
   absent field rather than failed retrieval. The probe rate appeared to fall to
   60% until inspection showed both "failures" had declined correctly in prose
   the refusal regex did not match. Neither number described anything the system
   did. The regex has since been fixed (see below) and the corrected rate is 5/5.
2. **The real cost of clause-blind chunking is 24 points of clause citation**,
   not the 97-point cliff `hit@k` implied. The model recovers clause numbers by
   reading them out of the window text, which is also why the citation axis fell
   only 0.47. But those numbers are model-extracted rather than carried as
   metadata, so nothing downstream can filter, verify or audit by clause.
3. **Completeness improved (+0.24), and that turned out not to be about
   chunking at all** - see Experiment 2.

## Experiment 2: top_k sweep (context volume)

Experiment 1 confounded chunk boundaries with context volume: at a fixed k=5,
1,887-char chunks feed the model 2.7x the text that 689-char chunks do. This
sweep holds clause-aware chunking fixed and varies k alone.

| Config           | Context | Cites clause | Complete | Relevant | Overall |
| ---------------- | ------- | ------------ | -------- | -------- | ------- |
| clause, k=5      | 3,446   | 97.1%        | 3.85     | 4.50     | 4.44    |
| clause, k=10     | 6,892   | 97.1%        | 4.09     | 4.44     | 4.46    |
| **clause, k=14** | 9,648   | **97.1%**    | **4.24** | 4.29     | **4.53** |
| sliding, k=5     | 9,433   | 73.5%        | 4.12     | 4.44     | 4.35    |

Three observations:

1. **The completeness gain was context volume, not chunk boundaries.**
   Completeness rises monotonically on unchanged chunking, 3.85 -> 4.09 -> 4.24,
   paired sign test p=0.013 (12 better, 2 worse, 20 unchanged). Well clear of the
   0.06 noise floor.
2. **At matched context volume, clause-aware wins outright.** k=14 (9,648 chars)
   beats the sliding window (9,433 chars) on completeness, 4.24 vs 4.12, while
   holding clause citation at 97.1% against its 73.5%. The stamped-metadata
   variant that Experiment 1 seemed to motivate has nothing left to win.
3. **Relevance is the cost, and it sets the ceiling.** It falls monotonically
   across the sweep, 4.50 -> 4.44 -> 4.29, as tangential excerpts dilute the
   answer. Context tokens also scale with k on both the answer and judge calls.
   k=14 is the knee, not a ceiling to raise without re-measuring.

`hit@1/3/5` held at 85/97/97% across all three, as expected - k does not reorder
the top 5.

## Experiment 3: retrieval strategy (vanilla vs hybrid vs rerank)

Three full runs at k=14, chunking and prompt held fixed. Hybrid adds BM25 to the
vector search and fuses with RRF; rerank takes hybrid's candidates and reorders
them with Voyage `rerank-2.5`.

```
uv run python evals/analyze.py compare_three \
  evals/results/20260804_162410_strategy_vanilla.csv \
  evals/results/20260804_163913_strategy_hybrid.csv \
  evals/results/20260804_165451_strategy_rerank.csv
```

| Metric           | Vanilla | Hybrid | Rerank |    H-V |    R-V |
| ---------------- | ------- | ------ | ------ | ------ | ------ |
| probe refusal %  |   100.0 |  100.0 |  100.0 |   +0.0 |   +0.0 |
| false refusal %  |     2.9 |    5.9 |    0.0 |   +2.9 |   -2.9 |
| hit@1 %          |    85.3 |   58.8 |   58.8 |  -26.5 |  -26.5 |
| hit@3 %          |    97.1 |   88.2 |   82.4 |   -8.8 |  -14.7 |
| hit@5 %          |    97.1 |   94.1 |   91.2 |   -2.9 |   -5.9 |
| cites expected % |    97.1 |   94.1 |   97.1 |   -2.9 |   +0.0 |
| citation         |    4.44 |   4.44 |   4.53 |  +0.00 |  +0.09 |
| grounding        |    4.88 |   4.74 |   4.85 |  -0.15 |  -0.03 |
| factual          |    4.71 |   4.59 |   4.71 |  -0.12 |  +0.00 |
| complete         |    4.24 |   4.15 |   4.21 |  -0.09 |  -0.03 |
| relevant         |    4.35 |   4.38 |   4.44 |  +0.03 |  +0.09 |
| **overall**      |    4.52 |   4.46 |   4.55 |  -0.06 |  +0.02 |

By difficulty: easy 4.78/4.58/4.85, medium 4.38/4.50/4.48, hard 4.42/4.29/4.31.

Four observations:

1. **Neither strategy separates on overall.** +0.02 for rerank and -0.06 for
   hybrid sit at or under the 0.06 judge noise floor established above. At n=34
   this experiment does not show that reranking helps; it shows it does not hurt.
   Read the per-question movements, not the mean.
2. **hit@1 falls 26.5 points without moving answer quality.** Both variants drop
   to 58.8% while `cites expected` holds at 97.1% for rerank. BM25 reorders the
   top of the list, but at k=14 the model still receives the right clause and
   still cites it - hit@1 is measuring rank, and rank is not what reaches the
   model. This is the clearest evidence yet that `hit@1` is the wrong headline
   metric at this k, and that hit@5 (97.1 -> 91.2) is the one worth watching.
3. **BM25 trades specific sub-clauses for their parents, and that is what the
   regressions are.** `management-review-h1` drops 4.80 -> 3.60, and the
   retrieved clauses show the mechanism directly:

   ```
   vanilla  2.1.3.5, 2.1.3.1, 2.1.2.2, 2.1.3.4, ...   specific sub-clauses
   hybrid   2.1.2,   2.1.3.1, 2.1.3.2, 2.1.3.5, ...   leads with a parent
   rerank   2.1.4, -, 2.1.2.1, 2.1.2.2, 2.1.4, -, ... parents + 3 chunks with no clause
   ```

   Section-level chunks repeat an element's vocabulary without stating any
   requirement, so BM25 scores them well and they displace the sub-clause that
   actually answers the question - `hit@3` goes False under rerank while
   vanilla had the answer at ranks 1 and 2. `verification-h1` shows the same
   shape (2.5.1.1 -> 2.5.1, 4.20 -> 3.20). The fix is chunk- or scoring-side:
   demote chunks whose clause is a parent of a more specific chunk in the same
   result set, or stop indexing section headers as retrievable chunks at all.

   A tokenizer hypothesis was tested here and rejected. `management-review-h1`
   is the only golden question naming a bare clause ("within Element 2.1"), and
   BM25 cannot match it: `2.1` is an atomic token, clause 2.1.3.5 tokenizes to
   `2.1.3.5`, and there is no prefix matching, so `2.1` reaches only 12 of 971
   chunks. Indexing clause parents alongside each token fixes exactly that - 12
   chunks reachable becomes 61, and the question's best expected-clause rank
   improves from 14 to 6 - but it moves BM25 clause hit@14 from 97.1% to 94.1%
   and leaves hit@28 unchanged. The questions it pushes out of the window
   contain no clause number at all; they lose because parent tokens let broad
   chunks outrank specific ones, which is the same defect above. Only 3 of 39
   golden questions name a clause number, so the reachable upside was small and
   the collateral cost was not. `retrievers/hybrid.py` records why it is absent.
4. **Rerank recovers what hybrid loses.** Hybrid alone is the worst of the three
   on overall, grounding, factual and false refusal (5.9%, its only 2/34 result);
   rerank consumes the same candidate set and returns to vanilla or better on
   every judged axis, plus the only 0% false-refusal run. The value is in the
   reordering, not in the extra recall.

Rerank costs two Voyage calls per question against one, so on the free tier's
3 RPM / 10K TPM it is roughly double the wall-clock and needs its candidate set
trimmed to a token budget. `retrievers/rerank.py` documents the measured budget.

## Resulting production configuration

Baseline, with `TOP_K` raised from 5 to 14 (`ask.py`). Chunking, embedding,
retrieval and prompt unchanged - Experiment 3 found no strategy change that
clears the noise floor, so vanilla retrieval stays in production. Measured:
overall 4.53, completeness 4.24, clause citation 97.1%, probe refusal 5/5,
false refusal 1/34.

## Defect found and fixed

`is_refusal` matched only the shape `<documents> do not <verb>`. It missed two
phrasings the model produces regularly:

```
"Based on the provided excerpts, there is no minimum number of CCPs required"
"Based on the provided excerpts, the specific dollar amount is not stated"
```

One probe alternated between matched and unmatched phrasings across runs of an
identical configuration, moving the reported refusal rate 20 points with no
change in behavior. The metric now requires a conjunction - the opening sentence
must both refer to the source documents and negate - rather than matching a list
of phrasings. Re-scored across all 5 runs, every run reads 5/5, and none of the
170 scored answers changed classification, so the added recall cost no precision.

The `refused` column in result CSVs written before this fix is stale, so
`analyze.py` no longer reads it: both derived metrics are recomputed from the
stored answer text on every load. They are pure functions of text the CSV
already holds, so this costs nothing and means a run scored under an older
metric reports correctly without being re-run. Every figure in this report comes
from that recomputation. `evals/check_metrics.py` pins the phrasings above so
the fix cannot silently rot.

## What this measured and what it didn't

This measures whether the system answers a small hand-curated set of SQF
questions usefully and cites the right clause, as judged by a strong LLM plus
deterministic clause matching. It does not measure: latency, cost, robustness
to adversarial phrasing, performance on documents outside the indexed set,
or whether the reference answers themselves are correct readings of the code.
The reference answers were written by one person from the documents and have
not been reviewed by a second practitioner. That is the largest single threat
to the validity of these numbers.

Two further limits specific to what is reported above. The probe set is n=5, so
the refusal rate moves in 20-point steps and cannot separate configurations at
this size - it is a smoke test, not an assurance figure. And `clause_hit@k` is
only comparable between configurations that chunk the same way; use the
answer-cites-expected-clause metric for anything else.

## What I'd do next

1. Second-reader review of the 34 reference answers and expected clauses
2. Larger golden set (100+) for more stable averages, and more than 5 probes
3. Stop BM25 returning section-level parents in place of the sub-clause that
   states the requirement (Experiment 3, observation 3), then re-run the
   strategy comparison. Hybrid and rerank were measured against the k=14
   baseline and neither cleared the noise floor, but this defect was working
   against both, so the comparison is not yet a fair one
4. Separate the retrieval metric from the generation metric fully, so a failure
   is attributable without reading the CSV. `answer_cites_expected_clause` is
   deliberately not that separation - it conflates retrieval with the model's
   willingness to cite, which is why `clause_hit_at_k` stays alongside it
5. Page-level citation for the 40 DOCX SOPs. `python-docx` has no pagination, so
   all 105 DOCX chunks carry `page=None` and their citations show source and
   clause but no page. The fix is upstream in ingest (render to PDF first), not
   in chunking
