# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A compliance RAG CLI over SQF food-safety certification documents. `README.md`
covers the corpus, the pipeline stages and the two design decisions that shape
everything (clause-aware chunking, refusal-first prompting) — read it first.
`evals/EVAL_REPORT.md` holds the measured results those decisions are justified
by, and `docs/` holds the week-by-week curriculum walkthroughs this repo is
being built against. Week 5 (LangChain/LCEL) is the current work.

Nearly every non-obvious choice in this codebase already has a comment
explaining what was tried and what it cost. Read the docstring before changing
a constant — several of them encode an eval result, not a guess.

## Commands

```bash
uv sync                                        # install
cp .env.example .env                           # then fill in keys

uv run python ingest.py                        # data/source/* -> data/raw/*.jsonl
uv run python chunk.py                         # data/raw/*     -> data/chunks.jsonl
uv run python embed.py                         # chunks         -> .chroma/ (rebuilds from scratch)

uv run python ask.py "How often are internal audits required?"
uv run uvicorn serve:app --reload --port 8000  # same answers over HTTP

uv run python evals/validate.py                # golden set schema + coverage; exit 1 on error
uv run python evals/check_metrics.py           # regression cases for metrics.py; exit 1 on failure
uv run python evals/run_eval.py NAME "notes"   # full 39-record run -> evals/results/*.csv
uv run python evals/run_eval.py smoke -n 3     # cheap subset
uv run python evals/analyze.py                 # summarize latest CSV
uv run python evals/analyze.py BASE.csv VAR.csv    # compare two runs
uv run python evals/analyze.py compare_three V.csv H.csv R.csv

uv run python langchain_rag.py build           # LangChain index -> .chroma_langchain/
uv run python langchain_rag.py                 # retrieval-only demo
uv run python langchain_rag.py ask "question"  # full LCEL chain
```

There is no pytest suite. `evals/check_metrics.py` and `evals/validate.py` are
the tests — both are plain scripts with meaningful exit codes, and both are
worth running after touching `evals/metrics.py` or `evals/golden.jsonl`.

## Rate limits shape the code

The Voyage account is on the free tier: **3 requests/minute, 10K tokens/minute**,
across embedding *and* rerank together. This is not a footnote — it is why
several modules look the way they do, and the most common way to break something
here is to add a Voyage call that bypasses the gate.

- `retrievers/embed.py` owns the process-wide gate. **Every** query-side Voyage
  call goes through `paced_call(operation, ...)`, which enforces one call per
  `VOYAGE_MIN_INTERVAL_SEC` (default 21s) and retries `RateLimitError` with
  backoff. `retrievers/rerank.py` and `langchain_rag.py` both route through it
  rather than constructing their own client.
- Never construct a bare `voyageai.Client()` for query-side work; pacing only
  works if it is the only door.
- `langchain_voyageai.VoyageAIEmbeddings` cannot be used directly — it builds
  its own client with `max_retries=0` and forbids extra constructor kwargs, so
  the first 429 propagates out of `retriever.invoke()`. `PacedVoyageEmbeddings`
  in `langchain_rag.py` exists for that reason.
- Ingest-side (`embed.py`) paces with its own `SLEEP_BETWEEN_BATCHES = 21`.
- Budget accordingly: a full 39-record eval is ~15 minutes on `vanilla` and
  longer on `rerank` (two Voyage calls per question). A full re-embed of 971
  chunks is ~45 minutes. Do not casually rebuild an index.
- `VOYAGE_MIN_INTERVAL_SEC=0` disables pacing once the account has billing.

## Architecture

### The clause metadata contract

`clause`, `clause_title`, `page` and `source` are the spine of this system. They
are set in `chunk.py`, stored as Chroma metadata by `embed.py`, returned in the
chunk dicts by every retriever, rendered into the `[Excerpt n | source | clause | p.page]` headers by `ask.assemble_prompt`, cited by the model, and then parsed
back out of the answer by `evals/metrics.py`. Drop a field anywhere on that path
and citation silently breaks at the far end — the answer still looks fine.

Chroma rejects `None` metadata values, so both indexers drop absent keys rather
than writing nulls. Chunks legitimately have no clause (front matter), so
`clause` is `.get()`-ed everywhere downstream, never indexed.

### Retrieval strategies

`ask.py` selects one at import time from `RETRIEVAL_STRATEGY` (`vanilla` |
`hybrid` | `rerank`) and wraps it in a single traced `retrieve()`. Everything
above that line — prompt assembly, generation, scoring, `serve.py`, the eval
harness — is strategy-agnostic. `hybrid` layers BM25 (with a clause-preserving
tokenizer) over `vanilla` via RRF; `rerank` layers Voyage rerank over `hybrid`.
The vanilla/hybrid/rerank comparison is settled (Week 5's LangGraph agentic loop is a separate axis, not a fourth strategy): **vanilla is production** (commit
`4bfd915`); the other two exist and are measured, and the reasoning is in
`EVAL_REPORT.md`.

### `ask.py` is the single source of truth

`SYSTEM_PROMPT`, `LLM_MODEL` (`claude-sonnet-4-6`) and `TOP_K` live there and
are imported by anything that needs them, including `langchain_rag.py`. `TOP_K`
defaults to 14 because of a 5/10/14 sweep documented at the constant — it is not
an arbitrary number, and a second copy of the prompt or a different k in another
module turns a pipeline comparison into a confounded one.

### Two indexes, deliberately separate

`.chroma/` (collection `sqf_docs`) is the raw-API index used by production.
`.chroma_langchain/` (collection `sqf_docs_lc`) is the LangChain one. Both are
built from the same `data/chunks.jsonl`; keeping them apart means Week 5
experiments cannot corrupt the index the eval baseline was measured on. Both are
gitignored and regenerable.

The LangChain build is resumable: chunk ids are stable and `langchain_chroma`
upserts, so an interrupted build skips what already landed.

### Configuration and tracing

`env.py` loads `.env` as an import-time side effect, with real environment
variables winning. `import env` must happen before any `os.environ` read —
that's why it sits with the imports in every entry point.

`tracing.py` builds the Langfuse client and is safe to import unconditionally:
with no `LANGFUSE_*` keys the SDK disables itself and every `@observe` becomes a
no-op. With *wrong* keys it prints one loud warning at startup and disables
tracing, and `evals/run_eval.py` refuses to start — deliberately, because a
long eval that finishes having sent nothing has happened before.

Spans are named and typed on purpose (`retriever`, `embedding`, `generation`),
so a trace shows where a question spent its time and what the judge scored.
Adding a pipeline path without tracing it makes it invisible in the comparison
the whole eval harness exists to support.

### Eval harness

`evals/golden.jsonl` is 34 scored questions plus 5 out-of-corpus refusal probes,
which have a different contract (`expect_refusal: true`, null `expected_clause`,
reference answer starting `Not found in the provided documents`) — `validate.py`
enforces the distinction. `run_eval.py` scores each record with deterministic
metrics plus a five-axis Claude judge, writes a timestamped CSV, and attaches
scores to Langfuse. `metrics.py` is pure functions over text and is the piece
with the tightest regression coverage; its clause matching is prefix-based in
both directions, which `check_metrics.py` pins down.
