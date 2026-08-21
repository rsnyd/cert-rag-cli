# TODO: align docs/walkthroughs with the real code

The code in this repo is the source of truth. The walkthroughs in
`docs/walkthroughs/` were written against an earlier state and lag it in the
places below. Each task is "bring the walkthrough's code block and surrounding
prose up to what the module actually does now" - not a code change. Read the
real module's docstrings first; several encode an eval result, not a guess.

Already aligned (do not redo): Voyage pacing (`paced_call`, token-budget
rerank), `env.py`/`.env` credential model, `tracing.py` auth check +
`TRACING_ENABLED`, `TOP_K=14`, `embed.py` `import env` + `clause_source`, the
resumable/paced LangChain build.

## Status: everything below is done (2026-08-21)

Every "complete file" code block in Weeks 1-6 is now a verbatim copy of the
module it names, verified by diff. When you change a module, update its block.
Re-verify with a diff of each block against its file before trusting a section.

## P1 - corpus identity (Weeks 1-2, Day 9) - DONE

- [x] Edition is **SQF Fundamentals 1.1 (FSC 19)**, not "Code Edition 9." The
      walkthrough's placeholder edition is wrong. Real `ingest.py` uses two tags:
      `_ED = "SQF Fundamentals 1.1"` for the certified corpus and
      `_ED_CODE10 = "SQF Code 10"` for the newer full code kept as reference, so
      answers do not conflate the two standards.
- [x] Replace the placeholder `MANIFEST` with the real one from `ingest.py` (the
      actual Spices Inc document set: the certification report, the FSMS manual,
      the two published standards, and the `2.x` / `11.x` SOP `.docx` files).
- [x] Add the tracked-changes extraction. Real `ingest.py` has `_element_text()`,
      which walks `<w:t>` descendants (via `docx.oxml.ns.qn`) so it picks up
      `<w:ins>` insertions that `python-docx`'s `.text` silently drops - that gap
      had removed an entire section of 2.6.3. The walkthrough still shows the
      naive `paragraph.text` loader.

## P1 - chunk.py rework (Weeks 1-2, Day 10) - DONE

- [x] Replace the simple clause chunker with the real one: two header forms
      (`_CLAUSE_INLINE` + `_CLAUSE_ALONE`), filename-clause fallback,
      cross-page threading via `document_lines`, ToC-leader stripping,
      `_looks_like_title` / `_inherited_title`, `clause_source`, repeated-id
      disambiguation, and the closing coverage stats.
- [x] Real counts throughout: 971 chunks (816 header / 87 filename / 68 none),
      and the ~45-minute embed at `BATCH=8` + 21s.

## P2 - metrics.py + check_metrics.py (Weeks 3-4, Day 2) - DONE

- [x] Segment-aware clause matching (`_segments` → integer tuples).
- [x] `parse_expected_clauses` with comma-list and range expansion.
- [x] Scoped-opening refusal detector (`_opening_sentence` + `_SCOPE_RE` AND
      `_NEGATION_RE`).
- [x] Reference-free metrics: `citation_grounding`, `cited_clauses`,
      `answer_cites_expected_clause`.
- [x] `check_metrics.py` added as a shown file, with the documented cases.

## P2 - validate.py + golden set (Weeks 3-4, Days 1-2) - DONE

- [x] Real `validate.py`: schema (missing *and* extra fields), `expect_refusal`
      ↔ `difficulty` consistency, corpus-grounding against `data/raw/`, the
      14-area `COVERAGE` map, collected-not-raised errors. Flat 30/10/10/10
      assertion dropped in favor of reporting the split.
- [x] Golden set shown at its real shape: 34 scored + 5 probes = 39, slug ids,
      real Fundamentals clauses, all three `expected_clause` forms, and real
      probe records (including the `<10 CFU` near-miss trap).
- [x] Real tag vocabulary and the four probe reason tags.

## P2 - ask.py + run_eval.py (Weeks 3-4, Day 4) - DONE

- [x] `_score_answer` added to the `ask.py` block, with why `refusal` is raw
      rather than pass/fail.
- [x] `run_eval.py`: `cites_expected` column, `dropped` tracking excluded from
      the denominator, `TRACING_ENABLED` refuse-to-start guard, repo-anchored
      `GOLDEN_FILE` / `RESULTS_DIR`.

## P3 - hybrid.py (Weeks 3-4, Day 10) - DONE

- [x] Fusion key is `_key = (source, page, full text)`, with the shared-prefix
      failure explained.
- [x] `k_per_retriever` defaults to `max(2k, 20)`; `CHUNKS_FILE` anchored to the
      repo root; FileNotFoundError-with-guidance for a fresh clone; the
      reverted clause-parent tokenizer experiment recorded.

## P3 - judge.py rubric refinements (Weeks 3-4, Day 3) - DONE

- [x] Sharpened rubric folded in, plus a "one defect, one axis" concept section
      explaining why.

## P3 - Week 5 query-side pacing (Weeks 5-6, Days 2-3) - DONE

- [x] `PacedVoyageEmbeddings` replaces the ungated `VoyageAIEmbeddings`, and the
      manual sleep-between-batches is gone.
- [x] The full LCEL chain imports `SYSTEM_PROMPT`, `LLM_MODEL`, `TOP_K`,
      `assemble_prompt` and `_score_answer` from `ask.py`; the chain carries
      `chunks` alongside `answer` instead of retrieving twice; `lru_cache` on
      `build_rag_chain` / `_chat_model`; `SystemMessage` rather than a
      `("system", ...)` tuple. Streaming example corrected to the generation
      sub-chain.

## P4 - serve.py - DONE

- [x] Written up as "Project: serve the RAG over HTTP" in **Weeks 5-6, Day 7**
      (the Document + Branch Strategy + Wrap day), alongside `FRAMEWORKS.md`.
      Weeks 10-11 turned out to be entirely about the Commerce AI Ops agent, so
      it would have landed there as an aside about a repo that week never
      touches. Covers the FastAPI wrapper, `X-API-Key` auth, the deduped
      `sources`, and why chunk text is not returned.

## Cross-cutting - DONE

- [x] Swept the later weeks for "Edition 9" / placeholder-clause references.
      Weeks 7-11 are a different project and never name a clause; the Weeks
      12-13 resume bullet said "30-question golden set" and now says 34 + 5.
- [x] Walkthrough figures kept clearly illustrative; the Day 5 sample output is
      labelled as such and points at `EVAL_REPORT.md` for real numbers.
- [x] `docs/README.md` file list rewritten (it named four `week-0N-*.md` files
      that do not exist) and `docs/NOTES.md` Observations filled in from what
      the code and `EVAL_REPORT.md` already record.
- [x] `README.md` and the Weeks 3-4 README block: `top_k=14`, 34 scored
      questions, and `validate.py` / `check_metrics.py` listed.

## Code changes made (the only ones)

- `evals/validate.py`: removed a stray `23` line from the module docstring, so
  the walkthrough's verbatim copy is not propagating a typo.
