"""Validate the SQF golden set: schema, refusal-probe convention, clause
grounding, and requirement-area coverage.

Run it before trusting the golden set in an eval:

    uv run python evals/validate.py            # validate evals/golden.jsonl
    uv run python evals/validate.py path.jsonl # validate another file

Exit code is 0 when the set is valid, 1 when any error is found. Warnings (for
example, a missing corpus so clause grounding is skipped) do not fail the run.

Scored questions and refusal probes have different contracts, keyed off
expect_refusal:
  - a scored question (expect_refusal false) names a real expected_clause;
  - a probe (expect_refusal true, difficulty "probe") has a null expected_clause
    and a reference_answer that starts with "Not found in the provided documents".
23
The set intentionally exceeds a flat 30/10/10/10 so it can cover every area in
COVERAGE at least once, so the difficulty split is reported, not asserted.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "evals" / "golden.jsonl"
RAW_DIR = REPO_ROOT / "data" / "raw"

SCHEMA_FIELDS = {"id", "difficulty", "tags", "question", "reference_answer",
                 "expected_topics", "expected_clause", "expect_refusal"}
DIFFICULTIES = {"easy", "medium", "hard", "probe"}
REFUSAL_PREFIX = "Not found in the provided documents"

_CLAUSE_TOKEN = re.compile(r"\b\d+(?:\.\d+){1,4}\b")

# Each requirement area maps to a predicate over a row. The set must hit every
# one at least once - this is the coverage checklist it is built to.
COVERAGE = {
    "Management commitment, policy, and management review":
        lambda r: {"system-elements", "management-review"} & set(r["tags"]),
    "Document control and record retention":
        lambda r: "document-control" in r["tags"],
    "Specifications and supplier approval":
        lambda r: "supplier-approval" in r["tags"],
    "The food safety plan and hazard analysis":
        lambda r: "food-safety-plan" in r["tags"] and not r["expect_refusal"],
    "CCP monitoring and critical limits":
        lambda r: "critical limits" in " ".join(r["expected_topics"]).lower(),
    "Corrective and preventative action":
        lambda r: "corrective-action" in r["tags"],
    "Verification and validation activities":
        lambda r: "verification" in r["tags"] and not r["expect_refusal"],
    "Internal audits":
        lambda r: "internal-audits" in r["tags"],
    "Product identification, traceability, withdrawal and recall":
        lambda r: "traceability-recall" in r["tags"],
    "Food defense and food fraud":
        lambda r: "food-defense" in r["tags"],
    "Allergen management":
        lambda r: "allergen-management" in r["tags"],
    "Training and competency":
        lambda r: "training" in r["tags"],
    "Calibration of monitoring equipment":
        lambda r: "calibration" in r["tags"],
    "Sanitation and pest control":
        lambda r: {"sanitation", "pest-control"} & set(r["tags"]),
}


def load_corpus_clauses() -> set[str] | None:
    """Set of clause tokens present in the corpus, or None if it is not here."""
    if not RAW_DIR.is_dir():
        return None
    present: set[str] = set()
    for fp in RAW_DIR.glob("*.jsonl"):
        for line in fp.open(encoding="utf-8"):
            present.update(_CLAUSE_TOKEN.findall(json.loads(line).get("text", "")))
    return present


def parse_clauses(expr: str) -> set[str]:
    """Clause tokens named by an expected_clause string.

    Handles comma-separated lists and hyphenated ranges; for a range only the
    two endpoints are returned (enough to confirm the range is real in-corpus).
    """
    tokens: set[str] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            tokens.update(p.strip() for p in part.split("-", 1))
        else:
            tokens.add(part)
    return tokens


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def validate_row(row: dict, present: set[str] | None) -> list[str]:
    """Return a list of error strings for one row (empty if the row is valid)."""
    errors: list[str] = []
    rid = row.get("id", "<no id>")

    missing = SCHEMA_FIELDS - set(row)
    extra = set(row) - SCHEMA_FIELDS
    if missing:
        errors.append(f"{rid}: missing fields {sorted(missing)}")
    if extra:
        errors.append(f"{rid}: unexpected fields {sorted(extra)}")
    # Fields below are read with .get(); missing ones are already reported above.

    difficulty = row.get("difficulty")
    if difficulty not in DIFFICULTIES:
        errors.append(f"{rid}: difficulty {difficulty!r} not in {sorted(DIFFICULTIES)}")

    if not (isinstance(row.get("tags"), list) and row.get("tags") and _is_str_list(row["tags"])):
        errors.append(f"{rid}: tags must be a non-empty list of strings")

    for field in ("question", "reference_answer"):
        if not (isinstance(row.get(field), str) and row.get(field, "").strip()):
            errors.append(f"{rid}: {field} must be a non-empty string")

    if not _is_str_list(row.get("expected_topics")):
        errors.append(f"{rid}: expected_topics must be a list of strings")

    refusal = row.get("expect_refusal")
    if not isinstance(refusal, bool):
        errors.append(f"{rid}: expect_refusal must be a boolean")
    # A probe is exactly a refusal row; the two signals must not disagree.
    if isinstance(refusal, bool) and refusal != (difficulty == "probe"):
        errors.append(f"{rid}: expect_refusal={refusal} disagrees with difficulty={difficulty!r}")

    ec = row.get("expected_clause")
    if refusal is True:
        if ec is not None:
            errors.append(f"{rid}: refusal rows must have expected_clause=null (got {ec!r})")
        if not str(row.get("reference_answer", "")).startswith(REFUSAL_PREFIX):
            errors.append(f'{rid}: refusal reference_answer must start with "{REFUSAL_PREFIX}"')
    elif refusal is False:
        if not (isinstance(ec, str) and ec.strip()):
            errors.append(f"{rid}: scored rows must cite a non-empty expected_clause")
        elif present is not None:
            absent = sorted(c for c in parse_clauses(ec) if c not in present)
            if absent:
                errors.append(f"{rid}: expected_clause not found in corpus: {absent}")

    return errors


def validate(path: Path) -> tuple[list[str], list[str], list[dict]]:
    """Validate a golden file. Returns (errors, warnings, rows)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        return [f"golden file not found: {path}"], warnings, []

    present = load_corpus_clauses()
    if present is None:
        warnings.append(f"corpus {RAW_DIR} not found - skipping clause grounding check")

    rows: list[dict] = []
    seen_ids: Counter[str] = Counter()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: invalid JSON ({e})")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {lineno}: expected a JSON object")
            continue
        rows.append(row)
        if isinstance(row.get("id"), str):
            seen_ids[row["id"]] += 1
        errors.extend(validate_row(row, present))

    for rid, count in seen_ids.items():
        if count > 1:
            errors.append(f"duplicate id {rid!r} appears {count} times")

    for area, hits in COVERAGE.items():
        try:
            if not any(hits(r) for r in rows):
                errors.append(f"coverage gap: no question covers '{area}'")
        except (KeyError, TypeError):
            # A malformed row already produced a schema error above; skip it here.
            pass

    return errors, warnings, rows


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_PATH
    errors, warnings, rows = validate(path)

    counts = Counter(r.get("difficulty") for r in rows)
    scored = sum(v for k, v in counts.items() if k != "probe")
    print(f"{path.name}: {len(rows)} rows | {scored} scored "
          f"(easy {counts['easy']}, medium {counts['medium']}, hard {counts['hard']}) "
          f"+ {counts['probe']} probes")

    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  FAIL  {e}")

    if errors:
        print(f"\nINVALID: {len(errors)} error(s)")
        sys.exit(1)
    print("\nVALID: schema, refusal convention, clause grounding, and 14-area coverage all pass")


if __name__ == "__main__":
    main()
