"""Deterministic metrics that need no judge call.

Two things are measured here rather than by the LLM judge, because both have a
crisp definition and we want them stable across runs:

  clause_hit_at_k  - did retrieval surface the clause that actually contains
                     the requirement, within the top k chunks?
  is_refusal       - did the system decline to answer?

Keeping these out of the judge makes them free, immune to judge drift, and
usable as the primary signal when comparing retrieval strategies in Week 4.
"""

# The exact phrase ask.py's system prompt instructs the model to use when the
# retrieved context does not contain the requirement. Keep the two in sync.
REFUSAL_MARKER = "not found in the provided documents"


def is_refusal(answer: str) -> bool:
    """True if the system declined to answer."""
    return REFUSAL_MARKER in answer.lower()


def clause_hit_at_k(chunks: list[dict], expected_clause: str, k: int) -> bool:
    """True if the expected clause appears among the top k retrieved chunks.

    Uses startswith so a chunk carrying 2.5.5.1 counts as a hit for an expected
    clause of 2.5.5 - a sub-clause of the right requirement is a correct
    retrieval, not a miss.
    """
    if not expected_clause:
        return False
    return any(
        (c.get("clause") or "").startswith(expected_clause)
        for c in chunks[:k]
    )


def retrieved_clauses(chunks: list[dict]) -> list[str]:
    """Clause numbers of the retrieved chunks, in rank order, for the CSV."""
    return [c.get("clause") or "-" for c in chunks]