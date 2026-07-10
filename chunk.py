"""Day 10 (SQF): Clause-aware chunking.

Reads the page-aware records from ingest.py (data/raw/*.jsonl) and emits one
chunk per clause to data/chunks.jsonl, carrying clause metadata. Same output
contract as a plain chunker (a jsonl of {id, source, text, ...}), just richer
and split on clauses instead of a fixed window.
"""
import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/chunks.jsonl")

MAX_CHARS = 2400   # sub-split a clause only if it exceeds this (~600 tokens)

# "2.4.3.1 Internal Audits" - requires at least one dot so plain list numbers
# ("1.") do not false-trigger. Tune to your documents' numbering scheme.
_CLAUSE = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s+(\S.*)$")


def split_into_clauses(text: str):
    """Yield [clause, title, body] lists, splitting on clause headers."""
    current = None
    for line in text.splitlines():
        m = _CLAUSE.match(line)
        if m:
            if current:
                yield current
            current = [m.group(1), m.group(2).strip(), line]
        elif current is not None:
            current[2] += "\n" + line
        else:
            current = [None, None, line]   # preamble before first clause
    if current:
        yield current


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
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for jf in sorted(RAW_DIR.glob("*.jsonl")):
            for line in jf.open(encoding="utf-8"):
                rec = json.loads(line)
                for clause, title, body in split_into_clauses(rec["text"]):
                    for i, piece in enumerate(sub_split(body.strip(), MAX_CHARS)):
                        if not piece:
                            continue
                        module = clause.split(".")[0] if clause else None
                        out.write(json.dumps({
                            "id": f"{Path(rec['source']).stem}__{clause or 'preamble'}__{rec['page']}__{i}",
                            "text": piece,
                            "source": rec["source"],
                            "doc_type": rec.get("doc_type"),
                            "edition": rec.get("edition"),
                            "module": module,
                            "clause": clause,
                            "clause_title": title,
                            "page": rec["page"],
                        }) + "\n")
                        written += 1
    print(f"Wrote {written} clause chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
    