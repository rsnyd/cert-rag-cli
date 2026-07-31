
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

## Resulting production configuration

Baseline, with `TOP_K` raised from 5 to 14 (`ask.py`). Chunking, embedding,
retrieval and prompt unchanged. Measured: overall 4.53, completeness 4.24,
clause citation 97.1%, probe refusal 5/5, false refusal 1/34.

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

The `refused` column in result CSVs written before this fix is stale; the
corrected figures in this report were recomputed from the stored answer text.

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
3. Hybrid retrieval and reranking (Week 4), now measurable against a k=14
   baseline rather than a k=5 one
4. Separate the retrieval metric from the generation metric fully, so a failure
   is attributable without reading the CSV
5. Promote answer-cites-expected-clause into `metrics.py` alongside
   `clause_hit_at_k`, since it is the metric that survives a chunking change
6. Regression test for the `is_refusal` phrasings above, so the fix does not
   silently rot
