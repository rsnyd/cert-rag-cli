# NOTES

Running notebook for `cert-rag-cli`. Tuning observations go under Observations as
I make them. Interview talking points go at the bottom.

## Observations

Measured results live in `evals/EVAL_REPORT.md` - baseline, the chunking
ablation, the TOP_K sweep, the three-way strategy comparison, and the judge
noise floor. This section is for the things that never became an experiment.

- **Clause regex.** One pattern was not enough. `_CLAUSE_INLINE`
  (`2.4.3.1 Internal Audits`) matches the SOPs and the audit report; both
  published standards put a bare `2.5.5` on its own line with the requirement
  below it, which needed `_CLAUSE_ALONE`. The SOPs state their clause only in
  the filename, hence the `filename_clause` fallback - 87 of 971 chunks.
- **Tracked changes in DOCX.** `python-docx`'s `Paragraph.text` reads only `w:r`
  runs that are direct children of the paragraph, so text inside `w:ins`
  (unaccepted insertions) is silently dropped. That removed an entire section of
  2.6.3. `ingest._element_text` walks `w:t` descendants instead. A silent partial
  extraction is the worst failure mode here - the answer still looks confident.
- **OCR never fired.** All four PDFs carry a native text layer at 1,400-3,500
  chars/page, so the `ocrmypdf` branch in `ingest.py` exists but has not run on
  this corpus.
- **Boilerplate `threshold`.** 0.6 catches the per-page version footers without
  eating real content.
- **Refusal detection is harder than a phrase match.** The model paraphrases the
  refusal about a fifth of the time, and a phrase-list matcher moved one probe's
  reported rate 20 points across identical runs. `metrics.is_refusal` now
  requires a conjunction (scope + negation) in the opening sentence.
  `evals/check_metrics.py` pins the phrasings.
- **TOP_K and refusal.** More context did not tempt the model into answering
  something it should refuse: across the 5/10/14 sweep the probe set stayed 5/5.

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
