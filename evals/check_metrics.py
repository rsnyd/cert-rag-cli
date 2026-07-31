"""Regression cases for the deterministic metrics in metrics.py.

Run it before trusting an eval, or after touching metrics.py:

    uv run python evals/check_metrics.py

Exit code is 0 when every case passes, 1 otherwise.

These metrics are pure functions over text, so they are cheap to pin down and
expensive to get wrong quietly: they carry the compliance headline for this
corpus, and a miss does not look like a bug, it looks like a worse system. The
refusal cases below are the specific phrasings that a phrase-list matcher missed
in production - one probe alternated between matched and unmatched wordings
across runs of an identical config and moved the reported rate 20 points.

Cases are written as (input, expected, why) so a failure says what broke rather
than just which index.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.metrics import (answer_cites_expected_clause, clause_hit_at_k,
                           is_refusal, parse_expected_clauses)

REFUSAL_CASES = [
    # The literal marker the system prompt asks for.
    ("Not found in the provided documents.", True, "literal marker"),
    ("**Not found in the provided documents.**", True, "marker behind markdown"),

    # Paraphrases with "documents" as the subject of the negation.
    ("The provided documents do not specify a minimum number of CCPs.",
     True, "documents-do-not paraphrase"),
    ("The excerpts do not provide the specific dollar amount of the annual fee.",
     True, "excerpts-do-not paraphrase"),

    # The two shapes a phrase-list matcher missed. Both decline; neither puts
    # "documents" as the subject of "do not". These are verbatim from probe runs.
    ("Based on the provided excerpts, there is no minimum number of Critical "
     "Control Points (CCPs) required for a compliant HACCP plan.",
     True, "scoped there-is-no"),
    ("Based on the provided excerpts, the specific dollar amount of the annual "
     "registration fee is not stated.",
     True, "scoped passive negation"),
    ("The specific annual fee amount is not found in the provided documents.",
     True, "trailing scope"),

    # Must NOT fire. A negation with no reference to the documents is ordinary
    # answer prose - this is the half of the conjunction that guards precision.
    ("Records must be retained for 2 years; there is no exemption for small sites.",
     False, "unscoped negation is a real answer"),
    ("The site must conduct internal audits annually (clause 2.5.5.1, p.28).",
     False, "plain answer"),
    ("The documents require a documented HACCP plan (clause 2.4.3, p.12).",
     False, "documents referenced without negation"),

    # Scoping to the opening sentence is what separates a refusal from an
    # answer that notes a gap after answering. Both contain decline language.
    ("Verification must be scheduled annually (clause 2.5.2, p.14). However, the "
     "excerpts do not specify who signs off.",
     False, "decline after answering is incomplete, not refused"),
]

CITES_CASES = [
    ("The site must audit annually (SQF Code, clause 2.5.5.1, p.28).",
     "2.5.5.1", True, "exact match"),
    ("Corrective action is required (clause 2.5.3.2, p.19).",
     "2.5.3", True, "sub-clause satisfies a parent expectation"),
    ("See clause 2.1.10 for details.",
     "2.1.1", False, "2.1.10 is not a child of 2.1.1"),
    ("Both 2.9.4.1 and 2.9.7.1 apply.",
     "2.9.4.1, 2.9.7.1", True, "comma list in expected_clause"),
    ("Clause 2.1.2.4 covers this.",
     "2.1.2.1-2.1.2.6", True, "range in expected_clause"),
    ("Not found in the provided documents.",
     "2.5.5.1", False, "refusal cites nothing"),
    ("The requirement is in clause 11.2.11.1.",
     "11.2.11", True, "two-digit segments"),
    ("Clause 2.5.5.1 applies.", "", False, "no expectation to satisfy"),
]

# hit@k reads chunk metadata; a chunker that emits no clause field scores 0
# regardless of what it retrieved. Pinned because that behavior is the whole
# reason answer_cites_expected_clause exists.
HIT_CASES = [
    ([{"clause": "2.5.5.1"}], "2.5.5", 1, True, "sub-clause hits parent"),
    ([{"clause": "2.1.10"}], "2.1.1", 1, False, "segment boundary respected"),
    ([{"clause": None}, {"clause": "2.5.5"}], "2.5.5", 1, False, "outside top k"),
    ([{"clause": None}, {"clause": "2.5.5"}], "2.5.5", 3, True, "inside top k"),
    ([{"clause": None}], "2.5.5", 5, False, "clause-blind chunk cannot hit"),
]


def main() -> None:
    failures = []

    for text, expected, why in REFUSAL_CASES:
        got = is_refusal(text)
        if got != expected:
            failures.append(f"is_refusal({why}): expected {expected}, got {got}\n"
                            f"      {text[:90]}")

    for answer, clause, expected, why in CITES_CASES:
        got = answer_cites_expected_clause(answer, clause)
        if got != expected:
            failures.append(
                f"answer_cites_expected_clause({why}): expected {expected}, got {got}\n"
                f"      answer={answer[:70]!r} expected_clause={clause!r}")

    for chunks, clause, k, expected, why in HIT_CASES:
        got = clause_hit_at_k(chunks, clause, k)
        if got != expected:
            failures.append(f"clause_hit_at_k({why}): expected {expected}, got {got}")

    # The parser underneath both clause metrics; 25 of 34 golden records use a
    # list or a range, so a silent regression here mis-scores most of the set.
    if parse_expected_clauses("2.1.2.1-2.1.2.6") != [(2, 1, 2, n) for n in range(1, 7)]:
        failures.append("parse_expected_clauses: range expansion broken")
    if parse_expected_clauses("2.1.2.1-6") != [(2, 1, 2, n) for n in range(1, 7)]:
        failures.append("parse_expected_clauses: bare-end range expansion broken")

    total = len(REFUSAL_CASES) + len(CITES_CASES) + len(HIT_CASES) + 2
    if failures:
        print(f"metrics: {len(failures)} of {total} cases FAILED\n")
        for f in failures:
            print(f"  FAIL  {f}")
        sys.exit(1)
    print(f"metrics: all {total} cases pass")


if __name__ == "__main__":
    main()
