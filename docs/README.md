# cert-rag-cli - curriculum walkthroughs (SQF corpus)

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

The delta lives almost entirely in ingestion plus a handful of corpus-specific
tweaks downstream:

- Week 2 (Days 9-10): PDF/DOCX extraction, running-header removal, OCR fallback
  for scanned files, tracked-change extraction from the Word SOPs, and
  clause-aware chunking with audit-grade metadata. This is a full rewrite of the
  loader/chunker, not a port.
- Week 3: SQF golden questions with expected clause references as ground truth,
  and a judge rubric that adds citation-correctness and no-fabrication
  dimensions. Tag Langfuse traces `corpus:sqf` so they do not mix with the
  frozen Drupal traces.
- Week 4 (Day 10): one tokenizer change so BM25 does not shred clause numbers
  like `2.4.3`. Optional metadata pre-filtering by module/doc_type.
- Week 4 (Day 12): same three-way comparison, but the headline metric is
  clause-citation accuracy, not just answer quality.

## A note on the week split

These week boundaries were inferred from what the frozen project contained. If
your original split placed Langfuse or the eval harness on different weeks, keep
your split and just lift the corpus-specific sections into the right file.
Nothing here depends on the exact boundary.

## Files

The walkthroughs live in `walkthroughs/`, split by week pair:

- `Weeks_1-2_Detailed_Walkthrough_SQF.md` - LLM API fundamentals, then RAG from
  scratch: `ingest.py` (PDF/DOCX, OCR fallback, tracked-change extraction),
  `chunk.py` (clause-aware chunking), `embed.py`, `ask.py`
- `Weeks_3-4_Detailed_Walkthrough_SQF.md` - the golden set and `validate.py`,
  deterministic metrics and `check_metrics.py`, the five-axis judge, the eval
  runner and `analyze.py`, Langfuse, then hybrid retrieval and reranking
- `Weeks_5-6_Detailed_Walkthrough_SQF.md` - LangChain/LCEL and LangGraph over
  the same clause chunks, evaluated on the same golden set; then fine-tuning,
  LoRA and QLoRA
- `Weeks_7-9_Detailed_Walkthrough_SQF.md` - agents (a separate project: the
  Commerce AI Ops agent, raw API then LangGraph)
- `Weeks_10-11_Detailed_Walkthrough_SQF.md` - cloud deployment and the
  certification sprint
- `Weeks_12-13_Detailed_Walkthrough_SQF.md` - portfolio, publishing, applying

Weeks 1-6 are this repo. Weeks 7-11 build a different project and only reference
this one.

## Keeping these honest

The code in this repo is the source of truth. Every "complete file" code block in
Weeks 1-6 is a verbatim copy of the module it names, so when you change a module,
update the block. Measured numbers belong in `evals/EVAL_REPORT.md` and
`docs/NOTES.md`; where a walkthrough shows a figure it should be clearly
illustrative, so it does not drift into contradicting the report.

## Portfolio angle

A compliance/regulatory RAG over food-safety certification documents is a
sharper portfolio piece than a generic docs bot. It shows domain-constrained
retrieval, audit-grade citation, and a refusal-to-fabricate posture that maps
directly to the "applied AI for content and e-commerce systems" specialization
you are pitching. Keep that framing in mind as you build; Week 4 has notes on
how to surface it.
