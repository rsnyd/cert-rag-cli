"""Deterministic metrics that need no judge call.

Two things are measured here rather than by the LLM judge, because both have a
crisp definition and we want them stable across runs:

  clause_hit_at_k  - did retrieval surface the clause that actually contains
                     the requirement, within the top k chunks?
  is_refusal       - did the system decline to answer?

Keeping these out of the judge makes them free, immune to judge drift, and
usable as the primary signal when comparing retrieval strategies in Week 4.
"""
import re

# The exact phrase ask.py's system prompt instructs the model to use when the
# retrieved context does not contain the requirement. Keep the two in sync.
REFUSAL_MARKER = "not found in the provided documents"

# The model does not always comply verbatim - it paraphrases roughly a fifth of
# the time - so matching only the literal marker scored correct refusals as
# failures.
#
# Enumerating whole phrasings does not hold up either. An earlier version of this
# module matched only "<documents> do not <verb>", which missed two shapes the
# corpus produces regularly:
#
#   "Based on the provided excerpts, there is no minimum number of CCPs required"
#   "Based on the provided excerpts, the specific dollar amount is not stated"
#
# Both decline; neither puts "documents" as the subject of "do not". One probe
# alternated between matched and unmatched phrasings across runs of an identical
# config, moving the reported refusal rate 20 points with no behavior change.
#
# So match a conjunction instead of a phrase list: the sentence must both refer
# to the source documents AND negate. Either half alone is common in a genuine
# answer ("the documents require ...", "there is no exemption for ..."), which is
# what keeps this from firing on answers that are merely discussing an absence.
_SCOPE_RE = re.compile(r"(?:provided\s+)?(?:documents?|excerpts?)", re.I)
_NEGATION_RE = re.compile(
    r"there\s+(?:is|are)\s+no\b"
    r"|\bis\s+not\s+(?:stated|specified|mentioned|provided|given|listed|found"
    r"|included|addressed|defined|established)"
    r"|\bdo(?:es)?\s+not\s+(?:specify|contain|provide|state|include|address"
    r"|mention|define|establish)",
    re.I,
)


def _opening_sentence(answer: str) -> str:
    """The answer's first sentence, with markdown emphasis and headings removed.

    Scoping the match to the opening is what separates a refusal from a partial
    answer. Both contain decline language, but only a refusal *leads* with it:
    an answer that cites requirements and then notes a gap ("However, the
    excerpts do not contain ...") is incomplete, not declined, and belongs to
    Completeness rather than to the refusal metrics.

    Known limitation: a multi-part question answered in part but opening with a
    decline for the other part reads as a refusal here. Distinguishing those
    needs semantics, which is exactly the judge drift this module avoids.
    """
    for line in answer.strip().splitlines():
        line = re.sub(r"[*_`]", "", line).strip()
        if not line or line.startswith("#") or set(line) <= {"-", "="}:
            continue
        return re.split(r"(?<=[.!?])\s", line, maxsplit=1)[0]
    return ""


def is_refusal(answer: str) -> bool:
    """True if the system's top-line response declined to answer."""
    opening = _opening_sentence(answer)
    if REFUSAL_MARKER in opening.lower():
        return True
    return bool(_SCOPE_RE.search(opening) and _NEGATION_RE.search(opening))


def _segments(clause: str) -> tuple[int, ...] | None:
    """Clause string to integer segments, or None if it is not a clause number."""
    parts = clause.strip().split(".")
    if not all(p.isdigit() for p in parts) or not parts[0]:
        return None
    return tuple(int(p) for p in parts)


def _expand_range(lo: str, hi: str) -> list[tuple[int, ...]]:
    """Expand '2.1.2.1'-'2.1.2.6' into every clause in between, inclusive.

    A range's endpoints must differ only in their last segment; the bare form
    ('2.1.2.1'-'6') is accepted too. Anything else is treated as two separate
    clauses rather than guessed at.
    """
    lo_segs, hi_segs = _segments(lo), _segments(hi)
    if lo_segs is None or hi_segs is None:
        return [s for s in (lo_segs, hi_segs) if s is not None]
    if len(hi_segs) == 1:                      # bare end: 2.1.2.1-6
        hi_segs = lo_segs[:-1] + hi_segs
    if lo_segs[:-1] != hi_segs[:-1] or hi_segs[-1] < lo_segs[-1]:
        return [lo_segs, hi_segs]
    prefix = lo_segs[:-1]
    return [prefix + (n,) for n in range(lo_segs[-1], hi_segs[-1] + 1)]


def parse_expected_clauses(expected_clause: str) -> list[tuple[int, ...]]:
    """Every clause named by an expected_clause field, as integer segments.

    The golden set writes this field three ways - a single clause ('2.1.1.1'), a
    comma-separated list ('2.9.4.1, 2.9.7.1'), and a range ('2.1.2.1-2.1.2.6') -
    and 25 of the 34 scored records use a list or a range. A metric that reads
    the field as one literal string scores those as misses no matter what
    retrieval returns, so parse it here rather than comparing raw text.
    """
    specs: list[tuple[int, ...]] = []
    for part in (expected_clause or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            specs.extend(_expand_range(lo, hi))
        elif (segs := _segments(part)) is not None:
            specs.append(segs)
    return specs


def clause_hit_at_k(chunks: list[dict], expected_clause: str, k: int) -> bool:
    """True if any expected clause appears among the top k retrieved chunks.

    Matching is on segment boundaries, so a chunk carrying 2.5.5.1 counts as a
    hit for an expected 2.5.5 - a sub-clause of the right requirement is a
    correct retrieval, not a miss - while 2.1.10 does not count as a hit for
    2.1.1. String prefixing conflates those two cases; the corpus has two-digit
    segments (11.2.11.1), so the distinction is real.
    """
    expected = parse_expected_clauses(expected_clause)
    if not expected:
        return False
    for c in chunks[:k]:
        got = _segments(c.get("clause") or "")
        if got is None:
            continue
        if any(got[:len(exp)] == exp for exp in expected):
            return True
    return False


def retrieved_clauses(chunks: list[dict]) -> list[str]:
    """Clause numbers of the retrieved chunks, in rank order, for the CSV."""
    return [c.get("clause") or "-" for c in chunks]