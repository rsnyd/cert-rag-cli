"""Clause-aware chunking.

Reads the page-aware records from ingest.py (data/raw/*.jsonl) and emits one
chunk per clause to data/chunks.jsonl, carrying clause metadata. Same output
contract as a plain chunker (a jsonl of {id, source, text, ...}), just richer
and split on clauses instead of a fixed window.

The clause number is load-bearing: it is what the model cites and what the eval
scores retrieval against, so a chunk that loses it is unciteable and can never
count as a hit. Three things have to go right for it to survive, and each was
dropping a different slice of the corpus:

  - the published standards put the clause number on a line of its own with the
    requirement text below it, so a pattern that expects "2.5.5 Internal Audits"
    on one line matches nothing in them;
  - a requirement that spans a page break continues on a page whose first line
    is not a header, so splitting page-by-page orphans the continuation;
  - the site SOPs carry no clause number in their body at all - it is in the
    filename ("2.8.1 Allergen Management For Food Fundamentals.docx").
"""
import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/chunks.jsonl")

MAX_CHARS = 2400        # sub-split a clause only if it exceeds this (~600 tokens)
HEADING_SEGMENTS = 3    # 2.1.2 is a heading with a title; 2.1.2.4 is a requirement under it
MAX_TITLE_CHARS = 80

# Two header forms. Inline ("2.4.3.1 Internal Audits") is what the SOPs and the
# audit report use; standalone is what both published standards use. Both
# require at least one dot so a plain list number ("1.") does not false-trigger.
_CLAUSE_INLINE = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s+(\S.*)$")
_CLAUSE_ALONE = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s*$")

# Table-of-contents rows ("Food Safety Policy ......... 24"). Left in, every ToC
# entry becomes a contentless clause chunk competing with the real requirement.
_TOC_LEADER = re.compile(r"\.{4,}")

# "i.", "ii.", "a)", "3." - list markers pymupdf emits on their own line.
_LIST_MARKER = re.compile(r"^\s*(?:[ivxlc]+|[a-z]|\d{1,2})[.)]\s*$", re.IGNORECASE)

# Leading clause number(s) in a filename stem, e.g. "2.5.1, 2.5.2 Validation...".
_FILENAME_CLAUSE = re.compile(r"^(\d+(?:\.\d+){1,4})")
_FILENAME_PREFIX = re.compile(r"^(?:\d+(?:\.\d+){1,4}[,\s]*)+")


def _looks_like_title(line: str) -> bool:
    """True if this line reads as a clause title rather than requirement prose.

    A title is short and self-contained; requirement text runs long and breaks
    mid-sentence, so it ends on a comma, colon or wrapped word.

    A clause number is explicitly not a title. Stripping ToC leader lines leaves
    each ToC entry's number sitting directly above the next one, and without this
    guard "2.1.2" adopts "2.1.3" as its title and every requirement beneath it
    inherits that instead of "Management Responsibility".
    """
    s = line.strip()
    if not s or len(s) > MAX_TITLE_CHARS:
        return False
    if _CLAUSE_ALONE.match(s) or _CLAUSE_INLINE.match(s):
        return False
    return s[-1] not in ".,;:" and not _LIST_MARKER.match(s)


def _inherited_title(titles: dict[str, str], clause: str) -> str | None:
    """Title of the nearest ancestor heading - 2.1.2.4 inherits from 2.1.2.

    Requirement clauses in the standards have no title of their own, so without
    this they carry clause_title=None and the citation header shows a bare number.
    """
    parts = clause.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:cut])
        if parent in titles:
            return titles[parent]
    return None


def filename_clause(source: str) -> tuple[str | None, str | None]:
    """(clause, title) named by a source filename, or (None, None).

    Used only as a fallback for documents whose body text carries no clause
    header - the SOPs are written PURPOSE / SCOPE / PROCEDURE with the clause
    they implement only in the filename.
    """
    stem = Path(source).stem
    m = _FILENAME_CLAUSE.match(stem)
    if not m:
        return None, None
    return m.group(1), _FILENAME_PREFIX.sub("", stem).strip() or None


def document_lines(recs: list[dict]):
    """Yield (page, line) for every line of a document, in reading order.

    Iterating the whole document rather than each page in isolation is what
    carries clause context across a page break.
    """
    for rec in recs:
        for line in rec["text"].splitlines():
            if not _TOC_LEADER.search(line):
                yield rec["page"], line


def split_into_clauses(lines):
    """Split a document's (page, line) stream into per-clause chunks.

    Yields dicts of {clause, clause_title, page, text, from_header}. from_header
    distinguishes a real clause header from preamble text, so contentless
    headings can be dropped without dropping unnumbered prose.
    """
    lines = list(lines)
    titles: dict[str, str] = {}
    chunks: list[dict] = []
    current: dict | None = None

    def start(clause, title, page, header_line, from_header):
        nonlocal current
        if current is not None:
            chunks.append(current)
        if clause and title:
            titles.setdefault(clause, title)
        current = {
            "clause": clause,
            "clause_title": title,
            "page": page,
            "body": [header_line] if header_line else [],
            "from_header": from_header,
        }

    i = 0
    while i < len(lines):
        page, line = lines[i]
        alone = _CLAUSE_ALONE.match(line)
        inline = _CLAUSE_INLINE.match(line)

        if alone:
            clause = alone.group(1)
            title, consumed = None, 1
            # Only heading-level numbers take the next line as a title; for a
            # requirement number that next line is the requirement itself.
            if (clause.count(".") + 1 <= HEADING_SEGMENTS
                    and i + 1 < len(lines) and _looks_like_title(lines[i + 1][1])):
                title, consumed = lines[i + 1][1].strip(), 2
            resolved = title or _inherited_title(titles, clause)
            header = f"{clause} {resolved}" if resolved else clause
            start(clause, resolved, page, header, True)
            i += consumed
            continue

        if inline:
            clause, title = inline.group(1), inline.group(2).strip()
            if not _looks_like_title(title):
                title = _inherited_title(titles, clause)
            start(clause, title, page, line.strip(), True)
            i += 1
            continue

        # Text with no clause header above it. Threading context across pages is
        # what preserves a clause number through a page break, but unnumbered
        # prose has no context worth preserving - and letting it run on would
        # merge a document's whole front matter into one chunk stamped with the
        # first page. Break those at the page boundary so the cited page is right.
        if current is None or (current["clause"] is None and page != current["page"]):
            start(None, None, page, line, False)
        else:
            current["body"].append(line)
        i += 1

    if current is not None:
        chunks.append(current)

    for ch in chunks:
        body = [ln for ln in ch["body"] if ln.strip()]
        # A clause header with nothing under it is a heading or a ToC remnant -
        # no requirement text, so nothing worth retrieving.
        if ch["from_header"] and len(body) <= 1:
            continue
        text = "\n".join(body).strip()
        if text:
            yield {**ch, "text": text}


def sub_split(body: str, size: int):
    """Only used when a single clause is very long. Keeps clause metadata intact."""
    if len(body) <= size:
        yield body
        return
    start = 0
    while start < len(body):
        yield body[start:start + size]
        start += size - 200   # small overlap


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    stats: Counter[str] = Counter()
    # Same clause number can appear more than once in a document (e.g. audit
    # reports cite a clause repeatedly), so disambiguate repeated base ids.
    seen_ids: dict[str, int] = {}
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for jf in sorted(RAW_DIR.glob("*.jsonl")):
            recs = [json.loads(line) for line in jf.open(encoding="utf-8")]
            if not recs:
                continue
            source = recs[0]["source"]
            fb_clause, fb_title = filename_clause(source)

            for ch in split_into_clauses(document_lines(recs)):
                clause, title = ch["clause"], ch["clause_title"]
                clause_source = "header" if clause else None
                if clause is None and fb_clause:
                    clause, clause_source = fb_clause, "filename"
                    title = title or fb_title

                for i, piece in enumerate(sub_split(ch["text"], MAX_CHARS)):
                    if not piece.strip():
                        continue
                    # Counted per written chunk, not per clause, so the reported
                    # coverage matches what actually lands in the corpus.
                    stats[clause_source or "none"] += 1
                    module = clause.split(".")[0] if clause else None
                    base_id = f"{Path(source).stem}__{clause or 'preamble'}__{ch['page']}__{i}"
                    n = seen_ids.get(base_id, 0)
                    seen_ids[base_id] = n + 1
                    chunk_id = base_id if n == 0 else f"{base_id}__{n}"
                    out.write(json.dumps({
                        "id": chunk_id,
                        "text": piece,
                        "source": source,
                        "doc_type": recs[0].get("doc_type"),
                        "edition": recs[0].get("edition"),
                        "module": module,
                        "clause": clause,
                        # How the clause was determined: a header in the text, or
                        # the filename. Keeps the weaker signal auditable.
                        "clause_source": clause_source,
                        "clause_title": title,
                        "page": ch["page"],
                    }) + "\n")
                    written += 1

    total = sum(stats.values())
    unclaused = stats["none"]
    print(f"Wrote {written} clause chunks to {OUT_FILE}")
    print(f"  clause from header:   {stats['header']}")
    print(f"  clause from filename: {stats['filename']}")
    print(f"  no clause:            {unclaused} ({100 * unclaused / max(total, 1):.0f}%)")


if __name__ == "__main__":
    main()
