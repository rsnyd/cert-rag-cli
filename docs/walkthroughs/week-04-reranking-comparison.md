# Week 4 - Reranking, comparison, and portfolio framing

## Objective

Add Voyage reranking on top of hybrid retrieval, run the full three-way
comparison (vanilla cosine, hybrid, hybrid+rerank) on the SQF golden set, and
package the result. The rerank call and the comparison harness port unchanged.
The corpus-specific work is deciding what "better" means for a compliance RAG and
framing the artifact for your pivot.

## Step 1 - Voyage rerank (port as-is)

Lift your rerank call from the Drupal build. It is corpus-agnostic.

```python
import voyageai

vo = voyageai.Client()

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    docs = [c["text"] for c in candidates]
    result = vo.rerank(query, docs, model="rerank-2", top_k=top_k)
    return [candidates[r.index] for r in result.results]
```

Feed it the fused candidate set from Week 3 (say top 20 from RRF), rerank down to
top 5, then generate. No pipeline change from Drupal beyond wiring it after RRF.

## Step 2 - where rerank earns its keep here

Rerank helps most when a query term is common across many clauses. SQF documents
repeat words like "records", "verification", "approved", "monitoring" in dozens
of clauses, so lexical and dense retrieval both return many near-duplicates that
differ only in which requirement they attach to. The cross-encoder reranker reads
the full query against each candidate and disambiguates. Expect the biggest lift
on broad-term queries ("what records must be kept for training") and little
change on already-precise ones ("clause 2.4.3").

## Step 3 - three-way comparison

Run the identical golden set across all three strategies and produce one table.
Do not lead with answer quality. Lead with the two metrics that matter for
compliance.

```
Strategy         | hit@3 | citation_correctness | grounding | refusal_rate | answer_quality
-----------------|-------|----------------------|-----------|--------------|---------------
vanilla cosine   |       |                      |           |              |
hybrid (RRF)     |       |                      |           |              |
hybrid + rerank  |       |                      |           |              |
```

Read it in this order:

1. `citation_correctness` and `grounding` - did retrieval improvements let the
   model cite the right clause and stop it fabricating requirements?
2. `refusal_rate` on the fabrication probe set - it must not drop as retrieval
   gets more aggressive. A common failure is that better retrieval surfaces a
   loosely-related clause that tempts the model to answer a question it should
   refuse. Watch for it.
3. `hit@3` - the underlying retrieval signal.
4. `answer_quality` last. In this domain it is the least load-bearing number.

## Step 4 - error analysis pass

Pull the rows where hybrid+rerank still misses. Categorize the causes; they tend
to cluster into a few buckets you can name:

- Requirement split across two clauses (chunk granularity, trace to Week 1).
- Clause number in the query not preserved in an index (tokenizer, trace to
  Week 3).
- Requirement lives in a table that extraction flattened badly (trace to Week 1;
  this is where you selectively bring in pdfplumber's table extraction).
- Genuine corpus gap (the requirement is not in your documents; refusal is
  correct and this is not an error).

Naming the failure taxonomy is itself a portfolio artifact. It shows you can
debug a retrieval system, not just assemble one.

## Step 5 - package it

- Freeze and tag the repo the way you did with `drupal-rag-cli`, so this becomes
  a second, domain-differentiated portfolio reference.
- Write the comparison table and the failure taxonomy into the project README.
- Keep the two screenshots: a correctly-cited answer next to its source PDF page,
  and a correct refusal on an out-of-corpus question.

## Portfolio framing for the pivot

This project reads differently from a generic docs bot, and that difference is
the point. When you describe it, foreground:

- Domain-constrained retrieval over messy real-world documents (scanned forms,
  tables, boilerplate), not a clean corpus.
- Audit-grade citation: every answer traces to a specific clause and page.
- A refusal-first posture measured with a dedicated probe set, not assumed.
- A named failure taxonomy from real error analysis.

That maps cleanly onto the applied-AI-for-content-and-compliance angle you are
pitching to the retrieval and eval companies on your target list. It also gives
you a concrete second data point next to the Drupal build, which demonstrates you
can re-target the same architecture at a new domain and reason about what
changes. That transfer story is exactly what an AI Solutions Architect interview
is probing for.

## Definition of done

- Three-way comparison table populated on the SQF golden set.
- Refusal rate on the probe set is stable or better under rerank.
- Named failure taxonomy in the README.
- Repo frozen and tagged as a portfolio reference.
