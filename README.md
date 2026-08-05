# cert-rag-cli

A retrieval-augmented question-answering CLI over SQF food-safety certification
documents. Ask a plain-English question about your certification program and get
an answer grounded in the actual policy, SOP, and audit-report text - with a
clause citation for every requirement, or an explicit "not found" when the
documents do not cover it.

This is a compliance/regulatory RAG: the corpus is a controlled body of
certification documents, and the job is to retrieve the governing requirement and
answer *only* from it. Getting a requirement subtly wrong in a food-safety
context is worse than not answering, so the system is built to cite or refuse
rather than fill gaps from general knowledge.

## Corpus

The SQF (Safe Quality Food) certification program for a single facility, as
`SQF Fundamentals 1.1 (FSC 19)`:

- **Policies** - food safety policy, reporting-structure statement
- **SOPs** - the numbered clause procedures (2.x management system, 11.x GMP)
- **Manual** - the Food Safety Management System PDF
- **Audit report** - the certification report

Mixed PDF and Word sources, some scanned. Documents are numbered by SQF clause
(e.g. `2.4.4`, `11.2.5`), and those clause numbers are the primary key an auditor
or practitioner reasons in - so they are carried through the whole pipeline as
metadata.

## How it works

The pipeline is four scripts, run in order. Each writes an intermediate file the
next one reads.

```
data/source/*.{pdf,docx}
      │
      ▼  ingest.py    extract pages, OCR scanned PDFs, strip running headers/footers
data/raw/*.jsonl      one {page, text, source, doc_type, edition} record per page
      │
      ▼  chunk.py     split on SQF clause headers, carry clause metadata
data/chunks.jsonl     {id, text, source, module, clause, clause_title, page, ...}
      │
      ▼  embed.py     Voyage embeddings -> local Chroma collection "sqf_docs"
.chroma/
      │
      ▼  ask.py       retrieve top-k, assemble cited context, answer with Claude
answer + clause citations
```

- **`ingest.py`** - PyMuPDF for PDFs, python-docx for Word (tables kept as
  pipe-joined rows). Scanned PDFs are detected by low text density and run through
  OCR (`ocrmypdf`). Running headers/footers and bare page numbers are removed by
  dropping lines that repeat across most pages. `doc_type` and `edition` come from
  a manifest, since they are not recoverable from the file itself.
- **`chunk.py`** - splits each document on clause headers (`2.4.3 Internal
  Audits`) rather than a fixed window, emitting one chunk per clause with its
  `clause`, `clause_title`, `module`, `source`, and `page`. Over-long clauses are
  sub-split with small overlap while keeping the clause metadata intact.
- **`embed.py`** - embeds chunks with Voyage (`voyage-3-lite`) into a persistent
  Chroma collection using cosine space. Batched and throttled for the free tier.
- **`ask.py`** - embeds the question, pulls the top-k chunks, formats them as
  labeled excerpts (`[Excerpt n | source | clause | p.page]`), and asks Claude to
  answer strictly from those excerpts with a clause citation per requirement.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Set the API keys the pipeline needs:

```bash
export VOYAGE_API_KEY=...      # embeddings (embed.py, ask.py)
export ANTHROPIC_API_KEY=...   # answer generation (ask.py)
```

Put your certification documents in `data/source/` and register each file in the
`MANIFEST` in `ingest.py` with its `doc_type`.

## Usage

Build the index once:

```bash
uv run python ingest.py     # data/source/*  -> data/raw/*.jsonl
uv run python chunk.py      # data/raw/*     -> data/chunks.jsonl
uv run python embed.py      # data/chunks.jsonl -> .chroma/
```

Then ask questions:

```bash
uv run python ask.py "What does our approved supplier program require?"
uv run python ask.py "How often do we calibrate metal detectors?"
```

Answers cite clauses as `(source, clause, p.page)`. If the retrieved excerpts do
not contain the requirement, the answer is `Not found in the provided documents`.

## Notes on choices

**Clause-aware chunking instead of a fixed window.** The obvious default is to
slice text into fixed-size overlapping windows. That is wrong for this corpus. An
SQF requirement is defined *by its clause* - "2.4.4 Approved Supplier Program" is
a single unit of meaning that an auditor cites as a unit. A fixed window would
routinely cut a requirement in half, or fuse the tail of one clause onto the head
of the next, so a retrieved chunk would carry a clause number in its metadata that
does not actually match the text inside it. `chunk.py` splits on clause headers
instead, so each chunk is one requirement and its `clause`/`clause_title` metadata
is trustworthy. That metadata is what makes citations verifiable and what enables
filtering by module or clause. (Clauses longer than ~600 tokens are sub-split as a
fallback, keeping the clause label attached.)

**A refusal-first system prompt because fabrication is the primary risk.** In a
general docs bot, a plausible-but-unsourced answer is a minor annoyance. In a
compliance setting it is the *main* failure mode: an answer that invents a
requirement, imports one from a different SQF edition, or states the right topic
with the wrong frequency or threshold can send a facility into a non-conforming
practice or a failed audit. So the system prompt in `ask.py` is built around
refusal, not helpfulness: answer only from the provided excerpts, cite the clause
for every requirement, and if the excerpts do not contain it, say "Not found in
the provided documents" and stop - never supply a requirement from general
knowledge or another standard. A confident "not found" is a correct answer here; a
confident fabrication is the one outcome the system is designed to prevent.

## Architecture

data/source/*.pdf,docx
  -> ingest.py     extract, OCR scanned pages, strip running headers
  -> data/raw/*.jsonl
  -> chunk.py      split on SQF clause boundaries, carry clause/page metadata
  -> data/chunks.jsonl
  -> embed.py      Voyage voyage-3-lite -> Chroma (sqf_docs)
  -> ask.py        retrieve -> assemble cited prompt -> Claude Sonnet 4.6

Retrieval strategies (RETRIEVAL_STRATEGY env var):
  vanilla  cosine similarity, top_k=5
  hybrid   BM25 + cosine fused with RRF (clause-safe tokenizer)
  rerank   hybrid candidates reranked by Voyage rerank-2.5

Evaluation:
  evals/golden.jsonl   34 scored questions + 5 out-of-corpus refusal probes
  evals/metrics.py     deterministic clause_hit@k and refusal detection
  evals/judge.py       Claude Sonnet 4.6, five axes including citation and grounding
  evals/run_eval.py    runner, CSV output, Langfuse scores
  evals/analyze.py     summarize / compare / compare_three

Observability: Langfuse, self-hosted. Every run is a session; every question is
a trace with judge scores attached.
