# cert-rag-cli - Weeks 1-4 walkthroughs (SQF corpus)

Reworked from the frozen `drupal-rag-cli` build. Same architecture, new corpus:
your company's SQF certification documents (PDFs and Word files) instead of the
Drupal Entity API docs.

## What is the same

You already ran this curriculum once, so these walkthroughs do not re-teach the
mechanics. Anywhere the code is corpus-agnostic, they say "port as-is" and point
you at the module in the frozen project. That covers the embedder, the vector
store, the RRF fusion, the Voyage rerank call, the LLM-as-judge scaffolding, and
the Langfuse client wiring.

## What is different

The delta lives almost entirely in Week 1 (ingestion) plus a handful of
corpus-specific tweaks downstream:

- Week 1: PDF/DOCX extraction, running-header removal, OCR fallback for scanned
  files, and clause-aware chunking with audit-grade metadata. This is a full
  rewrite of the loader/chunker, not a port.
- Week 2: SQF golden questions with expected clause references as ground truth,
  and a judge rubric that adds citation-correctness and no-fabrication
  dimensions. Tag Langfuse traces `corpus=sqf` so they do not mix with the
  frozen Drupal traces.
- Week 3: one tokenizer change so BM25 does not shred clause numbers like
  `2.4.3`. Optional metadata pre-filtering by module/doc_type.
- Week 4: same three-way comparison, but the headline metric is clause-citation
  accuracy, not just answer quality.

## A note on the week split

I inferred these week boundaries from what the frozen project contained. If your
original split placed Langfuse or the eval harness on different weeks, keep your
split and just lift the corpus-specific sections into the right file. Nothing
here depends on the exact boundary.

## Files

- `week-01-ingestion-baseline.md` - loaders, cleaning, clause chunking, baseline RAG loop
- `week-02-evaluation-observability.md` - SQF golden set, judge rubric, Langfuse
- `week-03-hybrid-retrieval.md` - BM25 + cosine via RRF, clause-safe tokenizer
- `week-04-reranking-comparison.md` - Voyage rerank, three-way comparison, portfolio framing

## Portfolio angle

A compliance/regulatory RAG over food-safety certification documents is a
sharper portfolio piece than a generic docs bot. It shows domain-constrained
retrieval, audit-grade citation, and a refusal-to-fabricate posture that maps
directly to the "applied AI for content and e-commerce systems" specialization
you are pitching. Keep that framing in mind as you build; Week 4 has notes on
how to surface it.
