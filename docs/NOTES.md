# NOTES

Running notebook for `cert-rag-cli`. Tuning observations go under Observations as
I make them. Interview talking points go at the bottom.

## Observations

(Empty so far. Record what you change and what it did to the eval numbers, the
same way the drupal-rag-cli notes tracked chunk size, TOP_K, and system prompt
changes.)

Things worth recording as you hit them:

- Clause regex `_CLAUSE` tuning: which numbering variants in the real documents
  did the first pattern miss.
- Boilerplate `threshold`: the value that actually caught the running footer
  without eating real content.
- Which source files tripped the OCR branch, and whether OCR output was usable.
- Chunking: clause-aware vs a fixed-window baseline, measured on clause_hit@3
  and citation correctness.
- TOP_K and the effect on refusal behavior (more context can tempt the model to
  answer something it should refuse).

## Interview talking points

Ingestion is the part of RAG that tutorials skip and real projects spend most of
their time on. Being able to say "the corpus was mixed native and scanned PDFs
with running headers and tabular requirements, so ingestion did extraction,
conditional OCR, and boilerplate detection before chunking" is worth more in an
SA interview than anything downstream.

Clause-aware chunking is a domain-specific engineering decision, not a generic
RAG step. A clause is the natural unit of an auditable requirement, so chunking
on clause boundaries is what makes an audit-grade citation possible at all.

Default tokenization shreds regulatory clause identifiers (`2.4.3` becomes `2`,
`4`, `3`), so lexical retrieval over a compliance corpus needs a clause-
preserving tokenizer. This is the kind of detail that signals retrieval built
over real documents rather than a tutorial corpus.

In a compliance setting a fluent answer citing the wrong clause is worse than a
refusal. That ordering drives the whole design: the refusal-first system prompt,
the fabrication probe set, and leading the eval table with citation correctness
and refusal rate rather than answer quality.
