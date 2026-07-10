# Week 3 - Hybrid retrieval (BM25 + cosine via RRF)

## Objective

Add lexical retrieval and fuse it with your dense retrieval using reciprocal
rank fusion, same as the Drupal build. The fusion math and the BM25 library port
unchanged. The one real edit is tokenization, because default tokenizers destroy
clause numbers, and clause numbers are exactly the kind of exact-match query a
compliance corpus attracts.

## Why BM25 matters more here

In the Drupal corpus, dense retrieval carried most of the load because queries
were conceptual. In the SQF corpus a large share of real queries are lexical and
precise: someone types "2.4.3", or "allergen", or "corrective action", or a
verbatim requirement phrase. BM25 nails exact terminology and clause references
that a dense embedder smears together. Expect hybrid to beat vanilla cosine by a
wider margin here than it did on Drupal, especially on your `document_control`
and `traceability_recall` categories.

## Step 1 - clause-safe tokenizer (the only real change)

A standard tokenizer splits on every non-alphanumeric character, so `2.4.3`
becomes `2`, `4`, `3` and the exact-match signal is gone. Preserve dotted
alphanumeric tokens.

```python
import re

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    # "Clause 2.4.3 internal audits" -> ["clause", "2.4.3", "internal", "audits"]
    return _TOKEN.findall(text.lower())
```

Use this same `tokenize` for both indexing and querying. If you added a
stopword list on the Drupal build, keep it, but do not stem clause tokens. A
stemmer that mangles `2.4.3` reintroduces the exact problem you just fixed.

## Step 2 - BM25 index (port, new tokenizer)

```python
from rank_bm25 import BM25Okapi

class LexicalIndex:
    def __init__(self, records: list[dict]):
        self.records = records
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in records])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
```

Consider indexing `clause + " " + clause_title + " " + text` rather than `text`
alone, so a query for "internal audits" matches the clause title even when the
body phrases it differently.

## Step 3 - reciprocal rank fusion (port unchanged)

No corpus-specific change. Standard RRF with k=60:

```python
def rrf(dense: list[str], lexical: list[str], k: int = 60, top_n: int = 8) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (dense, lexical):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_n]
```

Lift your existing implementation if it differs; do not rewrite it for SQF.

## Step 4 - optional metadata pre-filter

SQF queries often carry an implicit scope: "in the internal audit SOP", "under
Module 2". If you can detect a `doc_type` or `module` constraint, filter
candidates before fusion to lift precision. Keep it optional and off by default
so it does not mask retrieval weaknesses in your eval.

```python
def prefilter(records, *, doc_type=None, module=None):
    return [
        r for r in records
        if (doc_type is None or r["doc_type"] == doc_type)
        and (module is None or r["module"] == module)
    ]
```

## Step 5 - re-run the Week 2 eval

Run the identical golden set and rubric against hybrid. The number to watch is
`clause_hit_at_k`, broken down by category. You are looking for two things:

- Overall hit@3 up vs vanilla cosine.
- The lexical-heavy categories (document control, recall, exact-terminology
  requirements) improving the most.

Log both strategies to Langfuse with a `strategy` tag (`vanilla` vs `hybrid`) so
Week 4's three-way comparison is a query, not a re-run.

## Definition of done

- `tokenize("clause 2.4.3 audits")` yields `2.4.3` as a single token.
- Hybrid beats vanilla cosine on overall hit@3 on the SQF golden set.
- Both strategies' traces carry a `strategy` tag in Langfuse.

## Portfolio note

The clause-token insight is a genuinely domain-specific engineering decision, not
a generic RAG step. It is worth one sentence in your write-up: "default
tokenization shreds regulatory clause identifiers, so lexical retrieval needs a
clause-preserving tokenizer." That is the kind of detail that signals you have
built retrieval over real documents, not a tutorial corpus.
