# Week 1 - Corpus ingestion and baseline RAG

## Objective

Stand up the end-to-end baseline loop over the SQF corpus: load PDFs and Word
files, clean them, chunk them by clause, embed, store, retrieve with vanilla
cosine, and generate a cited answer. Same target as the Drupal build. The
difference is that your source is no longer clean markdown, so ingestion carries
almost all of the new work.

## What changes vs the Drupal build

| Concern | Drupal build | SQF build |
|---|---|---|
| Source format | Markdown / HTML | PDF + DOCX (some possibly scanned) |
| Structure | Headings already semantic | Clause numbering (`2.4.3.1`), modules |
| Noise | Minimal | Running headers, footers, page numbers |
| Tables | Rare | Common (audit matrices, checklists) |
| Chunk anchor | Markdown headers | Clause boundaries |
| Metadata | source, heading | source, doc_type, module, clause, clause_title, page, edition |
| Citation stakes | Low | High - a wrong clause is worse than "not found" |

Everything after "store" (embedder, vector store, retrieval math, generation
plumbing) ports directly. Spend your time on the loader and chunker.

## Step 0 - dependencies

```bash
cd ~/projects/cert-rag-cli
uv add pymupdf pdfplumber python-docx
# OCR fallback for scanned PDFs (optional branch, Step 2):
uv add ocrmypdf
# ocrmypdf needs system deps; on WSL2/Ubuntu:
sudo apt update && sudo apt install -y tesseract-ocr ghostscript qpdf poppler-utils
```

`pymupdf` (imported as `fitz`) is the fast primary extractor and gives you
per-page text. `pdfplumber` is slower but extracts tables cleanly; reach for it
only on files where tables carry the requirements. `python-docx` handles Word.

## Step 1 - loaders

`src/cert_rag/ingest/loaders.py`

```python
from pathlib import Path
import fitz  # pymupdf
from docx import Document as DocxDocument


def load_pdf(path: Path) -> list[dict]:
    """One record per page: {page, text}."""
    doc = fitz.open(path)
    try:
        return [
            {"page": i, "text": page.get_text("text")}
            for i, page in enumerate(doc, start=1)
        ]
    finally:
        doc.close()


def load_docx(path: Path) -> list[dict]:
    """One record per block. DOCX has no reliable page numbers, so page is None."""
    doc = DocxDocument(path)
    blocks: list[dict] = []
    for para in doc.paragraphs:
        if para.text.strip():
            blocks.append({"page": None, "style": para.style.name, "text": para.text})
    for t_idx, table in enumerate(doc.tables):
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                blocks.append({
                    "page": None, "style": "Table",
                    "text": " | ".join(cells), "table": t_idx,
                })
    return blocks
```

Note the DOCX page limitation up front. If a Word source is one you must cite by
page, convert it to PDF first (LibreOffice headless: `soffice --headless
--convert-to pdf file.docx`) and run it through the PDF path instead.

## Step 2 - OCR fallback for scanned PDFs

Real cert binders often contain scanned signed forms. Detect them and route to
OCR before extraction, otherwise you index empty pages.

```python
def is_probably_scanned(pages: list[dict], min_chars_per_page: int = 50) -> bool:
    if not pages:
        return False
    total = sum(len(p["text"].strip()) for p in pages)
    return total / len(pages) < min_chars_per_page
```

```python
import ocrmypdf

def ensure_text_layer(src: Path, out_dir: Path) -> Path:
    """OCR a scanned PDF into a searchable copy; return the path to use."""
    pages = load_pdf(src)
    if not is_probably_scanned(pages):
        return src
    dst = out_dir / f"{src.stem}.ocr.pdf"
    ocrmypdf.ocr(src, dst, skip_text=True, progress_bar=False)
    return dst
```

`skip_text=True` leaves already-textual pages alone and only OCRs the image
pages, so mixed documents are handled correctly. Re-run `load_pdf` on the
returned path.

## Step 3 - strip running headers, footers, page numbers

Boilerplate that repeats on most pages pollutes both embeddings and BM25. Detect
lines that recur across a large fraction of pages and drop them.

```python
import re
from collections import Counter

_PAGE_NUM = re.compile(r"^\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)


def strip_boilerplate(pages: list[dict], threshold: float = 0.6) -> list[dict]:
    counts: Counter[str] = Counter()
    for p in pages:
        for line in {ln.strip() for ln in p["text"].splitlines() if ln.strip()}:
            counts[line] += 1
    n = max(len(pages), 1)
    boiler = {line for line, c in counts.items() if c / n >= threshold}

    cleaned = []
    for p in pages:
        kept = [
            ln for ln in p["text"].splitlines()
            if ln.strip() not in boiler and not _PAGE_NUM.match(ln)
        ]
        cleaned.append({**p, "text": "\n".join(kept)})
    return cleaned
```

Tune `threshold` per document set. Version footers that appear on every page get
caught at 0.6; section headers that repeat only within one module will not.

## Step 4 - clause-aware chunking

This is the SQF-specific payoff. Chunk on clause boundaries so each chunk maps to
one auditable requirement, and carry the clause number and title as metadata.
That metadata is what turns a generic answer into an audit-grade citation like
"SQF Code Ed 9, clause 2.4.3.1, p.34".

```python
# Matches "2.4.3.1 Internal Audits" - requires at least one dot so plain
# list numbers like "1." do not false-trigger. Tune to your doc's numbering.
_CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s+(\S.*)$")


def split_into_clauses(text: str, page: int | None) -> list[dict]:
    chunks: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        m = _CLAUSE.match(line)
        if m:
            if current:
                chunks.append(current)
            current = {
                "clause": m.group(1),
                "clause_title": m.group(2).strip(),
                "page": page,
                "text": line,
            }
        elif current is not None:
            current["text"] += "\n" + line
        else:
            # preamble before the first clause on the page
            current = {"clause": None, "clause_title": None, "page": page, "text": line}
    if current:
        chunks.append(current)
    return chunks
```

Two guards on top of this:

- Oversized clause: if a clause chunk exceeds your token budget, sub-split it
  with your existing recursive splitter but copy the clause metadata onto every
  sub-chunk so citation survives.
- Undersized clause: do NOT merge across clause boundaries to hit a size floor.
  Blurring two requirements into one chunk is exactly the failure mode you are
  trying to avoid in a compliance corpus. A short chunk is fine.

## Step 5 - assemble records with metadata

`src/cert_rag/ingest/pipeline.py`

```python
def build_records(path: Path, *, doc_type: str, edition: str, ocr_dir: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        path = ensure_text_layer(path, ocr_dir)
        pages = strip_boilerplate(load_pdf(path))
        raw = [(pg["page"], pg["text"]) for pg in pages]
    elif path.suffix.lower() in {".docx", ".doc"}:
        raw = [(None, "\n".join(b["text"] for b in load_docx(path)))]
    else:
        return []

    records: list[dict] = []
    for page, text in raw:
        for i, ch in enumerate(split_into_clauses(text, page)):
            clause = ch["clause"]
            records.append({
                "id": f"{path.stem}__{clause or 'preamble'}__{page}__{i}",
                "text": ch["text"],
                "source_file": path.name,
                "doc_type": doc_type,               # sqf_code | sop | form | policy
                "module": clause.split(".")[0] if clause else None,
                "clause": clause,
                "clause_title": ch["clause_title"],
                "page": page,
                "edition": edition,                 # set to match YOUR documents
            })
    return records
```

`edition` is a plain field, not a hardcoded constant. Set it from whatever
edition your company's documents actually are (for example Edition 9). Do not
assume; the code should never bake in a specific edition.

Drive it from a small manifest so `doc_type` and `edition` are explicit per
file rather than guessed:

```python
# manifest.py or a YAML you load
MANIFEST = [
    {"path": "corpus/SQF_Food_Safety_Code.pdf", "doc_type": "sqf_code", "edition": "9"},
    {"path": "corpus/Internal_Audit_SOP.docx",  "doc_type": "sop",      "edition": "9"},
    # ...
]
```

## Step 6 - embed and store (port as-is)

No changes from the Drupal build. Lift your embedder and vector-store modules
directly. The only difference is that the metadata payload you persist alongside
each vector is now richer (clause, module, page, doc_type, edition). If your
store schema hardcoded the Drupal fields, widen it to carry these.

## Step 7 - baseline retrieval and generation

Retrieval (vanilla cosine top-k) ports unchanged. The generation prompt does
need updating, because the cost of a fabricated requirement is high here. Two
changes to your system prompt:

1. Require clause citation for every claim.
2. Require refusal when the retrieved context does not contain the requirement.

```text
You answer questions about the SQF certification documents provided in CONTEXT.

Rules:
- Answer only from CONTEXT. If CONTEXT does not contain the requirement, say
  "Not found in the provided documents" and stop. Never infer or supply a
  requirement from general knowledge.
- Cite the clause for every requirement you state, in the form
  (source_file, clause, p.page). If a chunk has no clause, cite the source_file
  and page.
- If clauses conflict or an answer spans multiple clauses, list each with its
  own citation.
```

Pass the metadata into the context so the model has clauses to cite. A minimal
context block per chunk:

```
[source_file=SQF_Food_Safety_Code.pdf clause=2.4.3.1 title="Internal Audits" p=34]
<chunk text>
```

## Definition of done

- `uv run cert-rag ingest` populates the store from the manifest with clause and
  page metadata visible on a sampled record.
- At least one known scanned form in the corpus produces non-empty chunks
  (OCR path verified).
- `uv run cert-rag ask "How often must internal audits be conducted?"` returns
  an answer that cites a real clause, and a deliberately out-of-scope question
  returns "Not found in the provided documents".
- Spot-check five chunks: each maps to a single clause, no two requirements
  merged into one chunk.

## Portfolio note

Take one screenshot now: a cited answer with a correct clause reference next to
the source PDF page. That single image is the most convincing artifact from the
whole build for a hiring conversation, because it demonstrates grounded,
citable, refusal-capable retrieval over messy real-world documents.
