# Weeks 3-4 Detailed Walkthrough (SQF corpus): Evaluation, Observability, Advanced Retrieval

**For the `cert-rag-cli` build. Week 3 builds the evaluation harness; Week 4 adds
observability and advanced retrieval. Every file is given complete and ready to
paste. Nothing here asks you to edit a file you already have; where a file
replaces an earlier version, the whole file is printed.**

By the end of Week 4 you have a compliance RAG with a 30-question golden set, an
LLM judge scoring five axes, deterministic clause-citation and refusal metrics,
full Langfuse tracing, and three retrieval strategies measured against each
other.

---

## Before You Start: Prerequisites Check (15 minutes)

You should be arriving from Week 2 with a working `cert-rag-cli`:
`ingest.py`, `chunk.py`, `embed.py`, `ask.py`, and `retrievers/` with
`vanilla.py` and `embed.py`. A question like "How often must internal audits be
conducted?" should return an answer citing a clause.

### 1. Install the new dependencies

```bash
cd ~/projects/cert-rag-cli
uv add pandas rank-bm25 langfuse
```

### 2. Keep eval results out of git

```bash
printf '%s\n' 'evals/results/*.csv' >> .gitignore
```

CSVs are regenerable and noisy in diffs. The summary tables you hand-copy into
`EVAL_REPORT.md` are what belongs in the repo.

### 3. Decide the corpus scope

Your golden set and your corpus have to agree. Pick a scope you can write 30
honest questions about and keep `data/source/` to the documents that support it.
A good first scope is the SQF System Elements module, which is dense,
self-contained, and maps to what an auditor actually asks about.

If you add or remove documents, rebuild the index from clean so the eval is
measuring the corpus you think it is:

```bash
rm -rf data/raw/* .chroma
uv run python ingest.py
uv run python chunk.py
uv run python embed.py
```

Sanity check before you build an eval on top of it:

```bash
uv run python ask.py "What must the SQF practitioner be responsible for?"
uv run python ask.py "What is the maximum fine for an OSHA violation?"
```

The first should cite a clause. The second should refuse. If the second answers
confidently, fix that before writing the golden set, because you are about to
spend two days building a measuring instrument and it should measure a system
that already behaves.

### Budget Note

Weeks 3-4 run roughly $5-8 of Anthropic and Voyage credits. The Week 4 three-way
comparison is the largest single spend at about $3. Each full 30-question eval
run is roughly $1 and 6-10 minutes.

---

# WEEK 3: Evaluation Infrastructure

The most important week of the plan. Most candidates pivoting to AI roles can
talk about RAG. Very few can talk credibly about evaluation. This is the gap.

---

## Day 1 (Mon): Why Evals Exist + Start the Golden Set

### Core Concept: Why Evals Exist

Traditional unit tests work because functions are deterministic. `add(2, 2)`
returns 4 every time. LLM outputs are not deterministic: the same prompt at
temperature 0 can produce slightly different text, the same retrieval pipeline
can return different chunks after a reindex, and different models phrase the same
correct answer differently.

You cannot regression-test an LLM system with `assert response == "expected
text"`. The expected text is a moving target. So you need four things: a
representative input set (the golden set), a scoring function that maps any
output to numbers, a baseline to compare against, and a workflow for running and
storing results. That is the eval loop. Teams without one ship a change, eyeball
a few outputs, and pray.

### Core Concept: LLM-as-Judge

You cannot write a Python function that scores "how good is this answer." You can
ask a strong LLM to do it. This works well with two caveats.

Use a model at least as strong as the one being judged. If your RAG uses Sonnet
4.6, judge with Sonnet 4.6 or stronger. The judge needs more reasoning headroom
than the generator, never less.

Score along multiple axes, not one. Each axis catches a different failure mode. A
confident wrong answer scores high on completeness and low on factual
correctness. An irrelevant tangent scores high on factual correctness and low on
relevance.

### Core Concept: What Changes for a Compliance Corpus

Two things make this eval different from a documentation-bot eval, and both are
worth understanding before you write a single question.

**Ground truth is a clause, not just prose.** Every SQF requirement lives at a
numbered clause. That gives you a second kind of ground truth: not only "what
should the answer say" but "which clause should have been retrieved." You can
measure that deterministically with no judge call at all, which makes it cheap,
stable, and immune to judge drift. This is your primary retrieval signal in Week
4.

**A wrong answer is worse than no answer.** In a compliance setting, a fluent
answer citing the wrong clause can send someone into an audit with a false belief.
A refusal costs them a search. That asymmetry drives three design decisions: two
extra judge axes (citation correctness and grounding), a set of deliberately
out-of-corpus probe questions where the correct behavior is refusal, and an eval
report that leads with citation accuracy rather than answer quality.

### Core Concept: The Golden Set

Thirty to fifty questions spanning what users will actually ask. Each entry has
the question, a reference answer, expected topics, tags, and for this build an
expected clause. You write these by hand. There is no shortcut. A bad golden set
produces meaningless scores no matter how good your judge is.

The common mistake is writing only easy questions where the right chunk is
obvious. Mix difficulties: easy (single clause, single lookup), medium (synthesis
across 2-3 clauses), hard (edge cases, comparisons, "what happens if"
reasoning).

On top of the scored questions, add 5 probe questions whose answers are
deliberately not in your corpus. These measure refusal, and they are the single
most important safety metric you will produce.

**What this repo's set actually became: 34 scored + 5 probes = 39 records**,
split easy 11 / medium 12 / hard 11. That is not a round 10/10/10, and the
lopsidedness is the point. Aim for a flat split and you will find yourself either
skipping a requirement area because you already have your ten hard questions, or
padding an area you have covered because you are one short. Coverage of the
requirement areas is the constraint that matters; the difficulty split is a
report, not a target. `validate.py` below is written that way - it *prints* the
split and *asserts* the coverage.

Ids are slugs, not counters: `internal-audits-m1`, not `Q017`. When a run fails
you read the id in a CSV row, and `sanitation-pest-m1` tells you where you are
without a lookup. Counters also invite renumbering every time you insert a
question, which quietly breaks comparisons against older result files.

### Reading (1 hour)

- Hamel Husain, "Your AI Product Needs Evals": https://hamel.dev/blog/posts/evals/
- Eugene Yan, "Evaluating LLMs is a minefield": https://eugeneyan.com/writing/evals/

### Project: Start the Golden Set (1 hour)

```bash
cd ~/projects/cert-rag-cli
mkdir -p evals/results
touch evals/golden.jsonl
```

Write five questions today. One JSON object per line. The schema:

| Field | Meaning |
|---|---|
| `id` | Area-and-difficulty slug: `internal-audits-m1`, `probe-air-changes-per-hour` |
| `difficulty` | `easy`, `medium`, `hard`, or `probe` |
| `tags` | 1-3 topic tags, used to slice results later |
| `question` | What the user asks |
| `expected_clause` | The clause(s) containing the requirement - single, comma list, or range. `null` for probes |
| `expect_refusal` | `true` only for probes |
| `reference_answer` | What a competent SQF practitioner would accept |
| `expected_topics` | 3-6 concrete terms that should appear |

> **Write these from your own documents.** The records below are the real ones
> from this repo's `golden.jsonl`, quoted verbatim so you can see the shape a
> finished record takes - the length of a reference answer, how specific
> `expected_topics` gets, how a probe explains *why* it is out of corpus. They
> are not a starter set to copy. SQF clause numbering differs by edition and by
> which code you are certified against (this corpus is Fundamentals 1.1, not
> Code Edition 9), and your internal SOPs have their own numbering entirely.
> Every `expected_clause` and `reference_answer` has to come out of the document
> in front of you, or the retrieval metric measures nothing and the judge scores
> against fiction.

Three scored records, one per `expected_clause` form:

```json
{"id": "sys-elements-e1", "difficulty": "easy", "tags": ["system-elements"], "question": "What must the site's food safety policy statement include at a minimum, and who is responsible for it?", "reference_answer": "Senior site management shall prepare and implement a policy statement (2.1.1.1) that outlines as a minimum: (i) the site's commitment to supply safe food; (ii) the methods used to comply with its customer and regulatory requirements; and (iii) the site's commitment to establish and review food safety objectives.", "expected_topics": ["food safety policy", "senior management", "safe food commitment", "customer and regulatory requirements", "food safety objectives"], "expected_clause": "2.1.1.1", "expect_refusal": false}
{"id": "internal-audits-e1", "difficulty": "easy", "tags": ["internal-audits"], "question": "How often must internal audits of the SQF System be conducted?", "reference_answer": "Per 2.5.5.1, the methods and responsibility for scheduling and conducting internal audits to verify the effectiveness of the SQF System shall be documented and implemented, and internal audits shall be conducted at least annually.", "expected_topics": ["internal audit", "at least annually", "verify SQF System effectiveness", "documented and implemented"], "expected_clause": "2.5.5.1", "expect_refusal": false}
{"id": "sys-elements-m1", "difficulty": "medium", "tags": ["system-elements"], "question": "Senior management must designate a person responsible for the SQF System at each site. What authority and qualifications must that person have?", "reference_answer": "Under 2.1.2.4 the designated person shall have responsibility and authority to: (i) lead the development and implementation of the GMPs in 2.4.2; (ii) oversee the development, implementation, review and maintenance of the SQF System; and (iii) take appropriate action to ensure the integrity of the SQF System. Under 2.1.2.5 that person shall be fully employed or contracted by the site, hold a position of responsibility in relation to managing the site's SQF System, be competent to implement and maintain food safety fundamentals, and have an understanding of the SQF Food Safety Fundamentals and the requirements to implement and maintain the SQF System for the site's scope.", "expected_topics": ["designated person", "authority", "GMP leadership", "SQF System integrity", "competency", "employed or contracted"], "expected_clause": "2.1.2.4, 2.1.2.5", "expect_refusal": false}
```

And the hard ones, where a single requirement genuinely spans a run of
sub-clauses, use a range:

```json
{"id": "sys-elements-h1", "difficulty": "hard", "tags": ["system-elements", "management-review"], "question": "How does SQF Fundamentals allocate management responsibility for food safety across reporting structure, resources, competency, and cover for absences?", "reference_answer": "Element 2.1.2 spreads the responsibility across six sub-clauses: 2.1.2.1 the reporting structure for those responsible for food safety shall be documented, identified and communicated within the site; 2.1.2.2 senior management shall ensure fundamental food safety practices and all applicable SQF System requirements are adopted and maintained; 2.1.2.3 senior management shall ensure adequate resources are available to achieve food safety objectives and support development, implementation, maintenance and ongoing improvement; 2.1.2.4 a designated person is appointed with authority over GMPs, oversight of the SQF System, and its integrity; 2.1.2.5 sets that person's competency and employment requirements; and 2.1.2.6 job descriptions for those responsible for food safety shall be documented and include provision to cover for the absence of key personnel.", "expected_topics": ["reporting structure", "adequate resources", "designated person", "competency", "job descriptions", "cover for absence"], "expected_clause": "2.1.2.1-2.1.2.6", "expect_refusal": false}
```

Those three forms - `2.1.1.1`, `2.1.2.4, 2.1.2.5`, `2.1.2.1-2.1.2.6` - are not
cosmetic. **25 of the 34 scored records use a list or a range**, so the metric
that reads this field has to parse it rather than string-compare it. That is
`parse_expected_clauses` on Day 2, and skipping it scores three-quarters of your
set as misses no matter how good retrieval is.

Two probe questions, to measure refusal from the first run:

```json
{"id": "probe-iso-22000-team-leader", "difficulty": "probe", "tags": ["probe", "cross-standard", "food-safety-plan"], "question": "What does ISO 22000 require for the qualifications of the food safety team leader?", "reference_answer": "Not found in the provided documents. ISO 22000 is a different standard and is not part of this corpus, which covers SQF Fundamentals 1.1 (and SQF Code Edition 10 for reference). A requirement from another standard must not be supplied. SQF Fundamentals does address HACCP training in 2.9.4.1, but the question asks specifically about ISO 22000, which the documents do not contain.", "expected_topics": ["Not found in the provided documents"], "expected_clause": null, "expect_refusal": true}
{"id": "probe-finished-product-micro-limit", "difficulty": "probe", "tags": ["probe", "not-specified", "supplier-approval"], "question": "What is the maximum total plate count and Salmonella limit that finished product must meet before it can be released?", "reference_answer": "Not found in the provided documents. The corpus does not state any numeric finished-product microbiological release limit. Clause 2.3.5.1 says finished product specifications may include microbiological and chemical limits but states no values, and 2.4.7.1 requires release only after inspections and analyses verify food safety controls are met, again without a numeric limit. The <10 CFU figure in the environmental monitoring program is an Enterobacteriaceae sanitation-indicator limit for surfaces, not a finished-product release specification, and must not be offered as one.", "expected_topics": ["Not found in the provided documents"], "expected_clause": null, "expect_refusal": true}
```

Note what makes a good probe: it sounds like it belongs. "What is the maximum
fine for jaywalking" is not a useful probe because no system would answer it. A
question about a neighbouring GFSI scheme, or a plausible-sounding threshold that
simply is not in your documents, is exactly the kind of thing a RAG will
confabulate when retrieval returns something adjacent.

The probe reference answers are longer than you would expect, and deliberately
so. A probe's reference answer has to name the *near miss* - the clause that
looks like an answer, and why it is not one. `probe-finished-product-micro-limit`
is the sharpest example: the corpus does contain a `<10 CFU` number, it is just
an environmental-surface sanitation indicator rather than a finished-product
release limit. Retrieval will surface it every time. Writing that trap into the
reference answer is what lets the judge tell "correctly refused" apart from
"refused because retrieval found nothing," which are very different systems.

Every probe's reference answer opens with the literal string
`Not found in the provided documents` - `validate.py` enforces that on Day 2, and
`is_refusal` in `metrics.py` keys off the same phrase.

```bash
git add evals/
git commit -m "Week 3 Day 1: eval scaffolding and first golden questions"
```

---

## Day 2 (Tue): Finish the Golden Set + Metrics

The slog day. The output is your most valuable artifact for the rest of the plan.

### Process

Block 2-3 uninterrupted hours. Write the remaining scored questions and the rest
of the probes in one sitting if you can. Work from your actual documents, open
beside you. Drive to coverage of every requirement area below, not to a
difficulty quota - this repo's set landed at 34 scored (easy 11, medium 12, hard
11) plus 5 probes because that is what covering all 14 areas took.

The natural rhythm is one area at a time, easy then medium then hard, which is
also where the id slugs come from: `training-e1`, `training-m1`, `training-h1`.
Some areas support all three, some only one - `calibration` has a single medium
question because there is one calibration requirement worth asking about.

Tag every question. This corpus's tag vocabulary, which the `COVERAGE` map in
`validate.py` keys off by exact string:

`system-elements`, `management-review`, `document-control`, `supplier-approval`,
`food-safety-plan`, `corrective-action`, `verification`, `internal-audits`,
`traceability-recall`, `food-defense`, `food-fraud`, `allergen-management`,
`training`, `calibration`, `sanitation`, `pest-control`

Probes additionally carry `probe` plus a reason tag naming *why* the answer is
absent: `not-specified` (the corpus is silent on a real question),
`cross-standard` (belongs to ISO 22000 or BRCGS, not SQF), `false-premise` (the
question assumes a requirement that does not exist, like a minimum CCP count),
`out-of-scope` (certification fees and other things outside the documents
entirely). Four different failure modes; you want at least one of each.

Keep the tags stable once you write them. They are strings in a `lambda` in
`validate.py`, so renaming `internal-audits` to `internal-audit` turns a coverage
assertion into a silent gap.

Tags let you slice results later: "where does this system fail? Mostly on
allergen management and calibration."

### Question-writing tips

- Avoid yes/no questions. "Does SQF require internal audits?" yields no
  information. "How often must internal audits be conducted and what is
  recorded?" does.
- Include realistic troubleshooting. "An internal audit found a CCP monitoring
  record with a gap. What does the code require me to do?" tests real value.
- Write reference answers as if explaining to a competent new QA coordinator.
  Specific enough to verify, not so specific that only one phrasing passes.
- Keep `expected_topics` to 3-6 concrete terms. They are checkboxes, not a
  thesaurus.
- Cover breadth. If 25 of 30 questions are about HACCP and 5 about everything
  else, you are testing the part of the corpus you find most interesting rather
  than the corpus.
- For `expected_clause`, use the most specific clause that actually contains the
  requirement. If a requirement genuinely spans several clauses, name them all -
  a comma list (`2.1.2.4, 2.1.2.5`) or, for a contiguous run, a range
  (`2.1.2.1-2.1.2.6`). Do not collapse to a parent clause to keep the field
  tidy: the metric already counts a retrieved sub-clause as a hit for its
  parent, so a parent is the *loose* answer, and writing `2.1.2` where you mean
  six specific sub-clauses makes the metric easier to pass than the question
  actually is.

### Coverage checklist

Hit at least one question in each area. These fourteen are exactly the `COVERAGE`
map in `validate.py`, which fails the run if any is unhit:

- Management commitment, policy, and management review
- Document control and record retention
- Specifications and supplier approval
- The food safety plan and hazard analysis
- CCP monitoring and critical limits
- Corrective and preventative action
- Verification and validation activities
- Internal audits
- Product identification, traceability, withdrawal and recall
- Food defense and food fraud
- Allergen management
- Training and competency
- Calibration of monitoring equipment
- Sanitation and pest control

### Project: `evals/validate.py`

This is one of the two things in this repo that function as tests (the other is
`check_metrics.py`, below). It does four jobs, and only the first is the obvious
one:

1. **Schema.** Every row carries exactly the eight fields - missing *and* extra
   are errors, so a typo'd key is caught rather than silently ignored by the
   runner's `.get()`.
2. **The probe convention.** `expect_refusal` and `difficulty == "probe"` must
   agree, a probe's `expected_clause` must be null, and its `reference_answer`
   must start with `Not found in the provided documents`. Three signals that all
   mean "probe", checked against each other, because a half-converted row scores
   as a failed scored question rather than a passed probe.
3. **Clause grounding.** Every `expected_clause` must actually appear somewhere
   in `data/raw/`. This is the check that earns its keep: it catches a clause
   number you transcribed wrong, or one that exists in Code Edition 10 but not
   in Fundamentals 1.1. Without it the metric reports a miss and you go looking
   at retrieval, when the bug is in the golden set. It downgrades to a warning
   when the corpus is not present, so a fresh clone can still validate schema.
4. **Coverage.** The fourteen requirement areas above, as predicates. Missing an
   area fails the run.

Note what it does *not* assert: a 30/10/10/10 split. It prints the distribution
and moves on, for the reason given on Day 1 - coverage is the constraint,
difficulty balance is a report. Exit code is 0 when valid, 1 on any error, so it
works in CI or a pre-commit hook.

Complete file. Create it and paste:

```python
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
```

```bash
uv run python evals/validate.py
```

Expected output:

```
golden.jsonl: 39 rows | 34 scored (easy 11, medium 12, hard 11) + 5 probes

VALID: schema, refusal convention, clause grounding, and 14-area coverage all pass
```

One structural note worth copying even if your corpus is different: errors are
*collected* and printed together, not raised on the first failure. An `assert`
that dies on row 3 hides the other eleven problems, so you fix one, re-run, find
the next, and spend an afternoon on what should have been one pass.

### Project: `evals/metrics.py`

The deterministic metrics. No LLM call, no cost, no drift. Four functions matter,
and each one is longer than the obvious version for a reason a run taught:

**Clause matching is on segments, not string prefixes.** `"2.1.10".startswith("2.1.1")`
is `True`, and this corpus has two-digit segments (`11.2.11.1`), so string
prefixing quietly counts wrong clauses as hits. `_segments()` turns a clause into
a tuple of ints and matching compares tuple slices, so `2.1.10` is not a child of
`2.1.1` while `2.5.5.1` still is a child of `2.5.5`.

**`expected_clause` has to be parsed.** 25 of the 34 scored records use a comma
list or a range, and comparing that field as one literal string scores every one
of them a miss. `parse_expected_clauses` handles all three forms, expanding
`2.1.2.1-2.1.2.6` into the six clauses it names (and accepting the bare-end form
`2.1.2.1-6`).

**Refusal is a conjunction, not a phrase list.** The model paraphrases the
refusal roughly a fifth of the time. Matching the literal marker alone scored
correct refusals as failures; enumerating phrasings did not hold up either,
because a phrase list built around "documents do not ..." missed real declines
like *"Based on the provided excerpts, there is no minimum number of CCPs
required."* One probe alternated between matched and unmatched wordings across
runs of an identical config and moved the reported refusal rate 20 points with no
behavior change - which is the kind of noise that makes an eval worse than
useless. The fix is to require two independent things in the *opening sentence*:
a reference to the source documents AND a negation. Either half alone is ordinary
answer prose. Scoping to the opening is what separates a refusal from an answer
that notes a gap after answering; the latter is incomplete, not declined.

**Two reference-free metrics.** `citation_grounding` and
`answer_cites_expected_clause` read the answer text rather than chunk metadata,
which buys two things. They work without a golden record, so `ask.py` can attach
them to any interactive question (Day 4). And they stay comparable across
chunkers: `clause_hit_at_k` reads `chunk["clause"]`, so a fixed-window ablation
that emits no clause field scores 0% no matter how good its retrieval is - it is
measuring the absent field. That same ablation cited the right clause in 73.5% of
its answers. Without a metric that reads the answer, you would have concluded
fixed-window retrieval was broken rather than merely unciteable.

Complete file:

```python
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


def cited_clauses(answer: str) -> list[tuple[int, ...]]:
    """Every distinct clause the answer cites, as integer segments, in order.

    Reads the answer with the same regex answer_cites_expected_clause uses, so
    the two agree on what counts as a citation.
    """
    seen: list[tuple[int, ...]] = []
    for match in _CLAUSE_IN_TEXT.finditer(answer or ""):
        segs = _segments(match.group(1))
        if segs and segs not in seen:
            seen.append(segs)
    return seen


def citation_grounding(answer: str, chunks: list[dict]) -> float | None:
    """Fraction of the answer's cited clauses that appear in the retrieved chunks.

    The reference-free companion to the judge's citation axis. It needs no golden
    record, so unlike the judged axes it can score an ad-hoc question, and it
    catches the failure that matters most in a compliance corpus: an answer
    citing a clause the model was never shown.

    Matching is prefix-based in BOTH directions, because either nesting is a real
    hit - an excerpt for 2.5.5.1 supports a citation of 2.5.5, and an excerpt for
    2.5.5 supports a citation of its sub-clause 2.5.5.1. That is looser than
    clause_hit_at_k, which only accepts the first direction because there the
    expected clause is authoritative and the chunk is what gets tested.

    Returns None when the answer cites nothing. That is an absence of evidence
    rather than a score of zero: a refusal correctly cites nothing, and scoring
    it 0.0 would drag the average down for behaving properly.
    """
    cited = cited_clauses(answer)
    if not cited:
        return None
    available = [segs for c in chunks
                 if (segs := _segments(c.get("clause") or "")) is not None]
    if not available:
        return 0.0
    supported = sum(
        1 for cit in cited
        if any(cit[:len(av)] == av or av[:len(cit)] == cit for av in available)
    )
    return supported / len(cited)


# Clause numbers as they appear in prose: "(source, clause 2.5.5.1, p.28)".
_CLAUSE_IN_TEXT = re.compile(r"\b(\d+(?:\.\d+){1,4})\b")


def answer_cites_expected_clause(answer: str, expected_clause: str) -> bool:
    """True if the answer text itself cites one of the expected clauses.

    The companion to clause_hit_at_k, and the one to reach for when comparing
    configurations that chunk differently. clause_hit_at_k reads chunk["clause"],
    so a chunker that does not emit that field scores 0% no matter how good its
    retrieval is - it measures the absent field, not the retrieval. A fixed-window
    ablation scored 0% on clause_hit_at_k while still citing the right clause in
    73.5% of answers, because the model reads clause numbers out of the chunk text.

    Reading the answer instead makes the metric chunk-size independent, at the
    cost of no longer isolating retrieval from generation: a correct clause that
    retrieval surfaced but the model declined to cite counts as a miss here. That
    is the right trade for a compliance corpus, where an uncited requirement is
    not a delivered answer.

    Matching is on segment boundaries, exactly as in clause_hit_at_k.
    """
    expected = parse_expected_clauses(expected_clause)
    if not expected:
        return False
    for match in _CLAUSE_IN_TEXT.finditer(answer or ""):
        got = _segments(match.group(1))
        if got and any(got[:len(exp)] == exp for exp in expected):
            return True
    return False
```

### Project: `evals/check_metrics.py`

`metrics.py` is pure functions over text, which makes it cheap to pin down and
expensive to get wrong quietly. A regression there does not raise; it just makes
your system look worse, or better, than it is - and you will spend the afternoon
looking at retrieval. This is the second of the two things that function as tests
in this repo, and it is worth writing the same day you write the metrics.

Cases are `(input, expected, why)` triples so a failure prints what broke rather
than an index. Every refusal case below is a phrasing that actually appeared in a
probe run, including the two that a phrase-list matcher missed and the three that
must *not* fire.

```python
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
```

```bash
uv run python evals/validate.py
# golden.jsonl: 39 rows | 34 scored (easy 11, medium 12, hard 11) + 5 probes
uv run python evals/check_metrics.py
# metrics: all 26 cases pass

git add evals/golden.jsonl evals/validate.py evals/metrics.py evals/check_metrics.py
git commit -m "Week 3 Day 2: 34-question SQF golden set, 5 refusal probes, deterministic metrics"
```

There is no pytest suite in this project, and these two scripts are why that is
defensible rather than lazy. Both are plain scripts with meaningful exit codes,
both run in under a second, and between them they cover the two things that can
silently corrupt every number downstream: a malformed golden set and a metric
that scores the wrong thing. Run both after touching `metrics.py` or
`golden.jsonl`.

---

## Day 3 (Wed): Tracing Setup + Build the Judge

### Why tracing comes first

The Langfuse SDK disables itself when its credentials are absent: every
`@observe` decorator becomes a no-op and your code runs exactly as before, just
untraced. That means you can write the instrumentation now, get it for free while
you build, and light it up in Week 4 by adding two environment variables and
changing no code at all.

This is a deliberate deviation from building the harness untraced and retrofitting
it later. It avoids writing every file twice.

### Credentials: `env.py` already exists

You built `env.py` in Week 2 Day 9 - the tiny module that loads the gitignored
`.env` at import time, with `import env` as the whole contract. `tracing.py`
below imports it, so the only setup step now is adding the Langfuse keys to
`.env` in Week 4 (Day 8), where you stand the server up. The `.env.example`
gains three lines then:

```bash
# added to .env in Week 4 Day 8
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

### Project: `tracing.py`

Complete file, at the repo root (not in `evals/`). It imports `env` so the
`LANGFUSE_*` keys resolve, and it checks the credentials once at startup rather
than discovering a bad key span by span during a 39-record run:

```python
"""Langfuse tracing setup - import this before any instrumented code runs.

Reads credentials from the environment, loaded from .env by the env module
(see .env.example for the template):

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000

If those keys are absent the Langfuse SDK disables itself: every @observe span
becomes a no-op and the app runs exactly as before, just untraced. So importing
this module is always safe, with or without a Langfuse account - which is why
the instrumentation can be written in Week 3 and switched on in Week 4.

If they are present but wrong, the credentials are checked once here rather
than discovered span by span, and TRACING_ENABLED says which of the three
states this process is in: keys absent, keys broken, or tracing live.

The decorated functions live in ask.py, evals/judge.py and evals/run_eval.py.
This module only owns client construction, so there is a single place that tags
every trace with the git commit (release). That is what lets you compare quality
across versions in the Langfuse UI.
"""
import logging
import os
import subprocess
import sys

import requests
from langfuse import get_client

# Loads .env into os.environ. Must come before the reads below, which is why it
# sits with the imports rather than inside a function.
import env  # noqa: F401

DEFAULT_HOST = "https://cloud.langfuse.com"


def _git_release() -> str | None:
    """Short commit SHA, recorded on every trace so runs are comparable across versions."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _auth_ok(host: str, public_key: str, secret_key: str) -> bool | None:
    """Do these credentials work? True/False, or None if the server was unreachable.

    Checked once at startup rather than left to the first span, because the SDK
    exports in a background batch: a bad key surfaces as a per-span 401 on
    stderr that no caller ever sees the return value of, so a long run happily
    finishes having sent nothing. One request here converts that into a single
    message before any work begins.
    """
    try:
        response = requests.get(
            f"{host.rstrip('/')}/api/public/projects",
            auth=(public_key, secret_key),
            timeout=5,
        )
    except requests.RequestException:
        return None
    return response.status_code == 200


_TRACING_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
AUTH_OK: bool | None = None

if _TRACING_ON:
    # The SDK reads release/environment from env vars at client construction, so
    # set sensible defaults before get_client() builds the process-wide
    # singleton. We don't overwrite values the user set explicitly.
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    _release = _git_release()
    if _release:
        os.environ.setdefault("LANGFUSE_RELEASE", _release)

    _host = os.environ.get("LANGFUSE_HOST") or DEFAULT_HOST
    AUTH_OK = _auth_ok(_host, os.environ["LANGFUSE_PUBLIC_KEY"],
                       os.environ.get("LANGFUSE_SECRET_KEY", ""))

    if AUTH_OK is not True:
        _reason = ("could not reach the server"
                   if AUTH_OK is None else "the server rejected the credentials")
        _key = os.environ["LANGFUSE_PUBLIC_KEY"]
        print(
            "\n" + "=" * 72 + "\n"
            f"LANGFUSE TRACING DISABLED - {_reason}.\n"
            f"  host: {_host}\n"
            f"  key:  {_key[:14]}...\n"
            "\n"
            "Everything else still runs; only tracing is off. Fix the LANGFUSE_*\n"
            "values in .env (see .env.example) and re-run. The secret key is shown\n"
            "once at creation and stored only as a hash - if it is lost, issue a\n"
            "new pair in the Langfuse UI under Settings -> API Keys.\n"
            + "=" * 72 + "\n",
            file=sys.stderr,
        )
        # Switch the SDK off outright instead of letting it retry every span.
        # Read at construction, so it has to be set before get_client() below.
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logging.getLogger("langfuse").setLevel(logging.CRITICAL)
else:
    # No credentials: the SDK still runs but every span/update would otherwise
    # log an auth/no-span warning per call. Silence that so untraced runs stay
    # quiet - the @observe spans become harmless no-ops.
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)

# get_client() builds (or returns) the singleton from the LANGFUSE_* env vars.
langfuse = get_client()

# True only when credentials are present AND the server accepted them. Callers
# that exist to produce traces (evals/run_eval.py) refuse to start without it.
TRACING_ENABLED = _TRACING_ON and AUTH_OK is True

__all__ = ["langfuse", "TRACING_ENABLED"]
```

### Concept: The Judge Prompt Is Engineered, Not Written

A bad judge prompt is "score this answer from 1 to 10." A good one does five
things: defines each axis with a clear rubric, gives concrete anchors for high
and low scores, provides the reference answer as a comparison, asks for brief
reasoning before scoring, and returns structured output your code can parse.

You get the structured output using the forced-tool-use pattern from Week 1 Day
5: define a tool whose `input_schema` is the score shape and set `tool_choice` to
force it.

### Concept: Why five axes here

The three standard axes (factual correctness, completeness, relevance) measure
whether the prose is good. For a compliance corpus you need two more.

**Citation correctness** asks whether the answer cited the clause that actually
contains the requirement. An answer can be word-perfect and still send someone to
the wrong clause. In an audit that is a real failure, and none of the three
standard axes catch it.

**Grounding** asks whether every claim is supported by the retrieved context. This
is the fabrication axis. A model that invents a plausible frequency ("verified
quarterly") when the context says nothing about frequency scores low here even if
the rest is fine.

To score those two, the judge needs to see the retrieved context, not just the
answer. That is why `judge()` below takes a `chunks` argument.

### Concept: one defect, one axis

Five axes only tell you where a problem is if each defect is deducted exactly
once. Left to itself the judge double-counts: a wrong clause number reads as a
Citation defect *and* a Grounding defect *and* a Factual defect, so one mistake
moves three axes and the per-axis breakdown stops locating anything. The rubric
below assigns ownership explicitly - a wrong clause or page belongs to Citation
Correctness alone, omitted key points to Completeness alone - and asks the judge
to re-read its own scores and strip duplicated deductions before submitting.

Two related sharpenings, each of which moved real scores. Completeness now
requires listing the reference's key points *before* counting them, because a
judge scanning for general thoroughness rewards a long confident answer that
covers half the reference. And Grounding now scores claim *content* only, with
citation labels explicitly out of scope, since a correctly-supported requirement
attached to the wrong clause number is one defect, not two.

### Project: `evals/judge.py`

Complete file:

```python
"""LLM-as-judge for SQF RAG output scoring."""
import json
import sys
from pathlib import Path

from anthropic import Anthropic
from langfuse import observe

# Make the repo root importable so `tracing` resolves when judge.py is run
# directly or imported from the evals package.
sys.path.insert(0, str(Path(__file__).parent.parent))
from tracing import langfuse

JUDGE_MODEL = "claude-sonnet-4-6"

JUDGE_SYSTEM = """You are an expert evaluator scoring an AI assistant's answer to a
question about SQF food-safety certification requirements.

You will receive:
- The question that was asked
- A reference answer (what a competent SQF practitioner would consider correct)
- The retrieved document excerpts the assistant was given as context
- The actual answer the assistant produced

Score the answer on five axes, each from 1 to 5. The axes are independent: score
each one on its own and do not let a low score on one drag down the others. Judge
the answer as a response to THIS question - a statement that is true in isolation
but says nothing about what was asked earns no credit.

FACTUAL CORRECTNESS - are the substantive claims the answer actually makes correct?
Judge only the claims present. Missing points are handled under Completeness, a
wrong clause number or page is handled under Citation Correctness, and a
terse-but-correct answer still scores 5 here.
  5 = Every claim is accurate. No false or misleading statements.
  4 = Accurate apart from one minor imprecision that would not mislead a practitioner.
  3 = Mostly correct but contains one significant inaccuracy someone could act on wrongly.
  2 = Multiple significant inaccuracies, or a central claim is wrong.
  1 = The core requirement is wrong, or a frequency, threshold or responsibility is
      misstated in a way that would fail an audit (a program described as quarterly
      when the requirement is annual).

COMPLETENESS - how many of the reference answer's key points are covered?
Count only content that matches a key point in the reference. Material the
reference does not ask for never raises this score, however detailed or
confident it is, and an invented requirement never substitutes for a key point
the answer omitted. First list the reference's key points, then count how many
the answer covers, then pick the band.
  5 = Covers every key point in the reference (extra correct detail is fine).
  4 = Covers all but one key point.
  3 = Covers roughly half the key points.
  2 = Covers at most one key point; misses the rest.
  1 = Addresses none of the reference's key points.

RELEVANCE - is the answer on-topic for the question?
  5 = Entirely on-topic; every sentence bears on the question.
  4 = On-topic with one minor tangent.
  3 = On-topic but padded or verbose with filler.
  2 = Roughly half the content is off-topic, or it answers a different question.
  1 = Does not engage the question at all.

CITATION CORRECTNESS - does the answer cite the clause that actually contains the
requirement it states? Check the cited clause against the excerpts.
  5 = Every requirement stated is cited, and the cited clause is the one that
      contains it.
  4 = Correctly cited apart from one missing page or source detail.
  3 = Cites a clause in the right area but not the one holding the requirement,
      or cites some requirements and not others.
  2 = Cites a clause that does not support the claim attached to it.
  1 = Cites a clause that does not appear in the excerpts at all, or states
      requirements with no citation whatsoever.
A confident, well-written answer that cites the wrong clause scores 1-2 here. Do
not let good prose lift this score.

GROUNDING - is every substantive claim supported by the excerpts, with nothing
invented? Score the content of the claims only. Citation labels - clause numbers,
page numbers, source filenames - are not claims on this axis. A requirement the
excerpts do support scores 5 here even when it is attached to the wrong clause
reference; pointing at the wrong clause is a Citation Correctness defect and must
not be deducted twice.
  5 = Every claim traces to the excerpts. Nothing added from outside knowledge.
  4 = Fully supported apart from one harmless general statement.
  3 = Mostly supported, but one claim goes beyond what the excerpts say.
  2 = Contains a fabricated specific: a frequency, threshold, temperature,
      responsibility or record requirement that appears nowhere in the excerpts.
  1 = Substantially invented, or answers from general food-safety knowledge when
      the excerpts do not contain the requirement.
Any fabricated specific caps this axis at 2 regardless of how much else is right.

Calibration:
  - Each defect is scored once, on the axis that owns it. A wrong clause or page
    is a Citation Correctness defect only. Omitted key points are a Completeness
    defect only. An invented requirement is a Grounding and Factual Correctness
    defect. Before submitting, re-read your scores and remove any deduction you
    made on an axis for a defect another axis already owns.
  - A fully correct answer MUST score 5 on the axes it satisfies. Do not withhold
    5 to seem strict, and do not force a 3-4 "average" - accuracy matters, not a
    target distribution.
  - A genuinely wrong or off-topic answer MUST score 1-2 on the axis it fails. Do
    not soften an answer that cites the wrong clause or invents a requirement.
  - If the assistant declined to answer and the excerpts genuinely do not contain
    the requirement, that is correct behavior: score citation_correctness and
    grounding 5, and score the other three axes against what the reference asks for.

Before scoring, write 1-3 sentences of reasoning about strengths and weaknesses.
Use plain hyphens, never em dashes."""

SCORE_TOOL = {
    "name": "score_answer",
    "description": "Submit scores for the assistant's answer along five axes with brief reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "1-3 sentences explaining strengths and weaknesses before scoring."
            },
            "factual_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
            "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "citation_correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": [
            "reasoning", "factual_correctness", "completeness",
            "relevance", "citation_correctness", "grounding",
        ]
    }
}

_client = Anthropic()


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks the same way ask.py shows them to the model."""
    if not chunks:
        return "(no excerpts were retrieved)"
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[Excerpt {i} | {c.get('source', 'unknown')}"
        if c.get("clause"):
            header += f" | clause {c['clause']}"
        if c.get("page") is not None:
            header += f" | p.{c['page']}"
        header += "]"
        blocks.append(f"{header}\n{c.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


@observe(name="judge", as_type="generation")
def judge(question: str, reference_answer: str, system_answer: str,
          chunks: list[dict]) -> dict:
    """Score a single answer against its reference and its retrieved context.

    Traced as a generation so the judge's own model and token cost is tracked
    separately from the answer it grades. When run inside an eval item (see
    run_eval.py) this nests under the same trace as the answer being judged.
    """
    user_prompt = f"""QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED EXCERPTS THE ASSISTANT WAS GIVEN:
{_format_context(chunks)}

ASSISTANT'S ANSWER:
{system_answer}

Score the assistant's answer using the score_answer tool."""

    response = _client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=JUDGE_SYSTEM,
        tools=[SCORE_TOOL],
        tool_choice={"type": "tool", "name": "score_answer"},
        messages=[{"role": "user", "content": user_prompt}]
    )

    for block in response.content:
        if block.type == "tool_use":
            langfuse.update_current_generation(
                model=JUDGE_MODEL,
                input=user_prompt,
                output=block.input,
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            )
            return block.input

    raise RuntimeError("Judge did not return a tool_use block")


if __name__ == "__main__":
    # Calibration checks. Run these once and read the scores - if the judge does
    # not separate these three cases cleanly, tighten the rubric before you
    # trust a full run.
    context = [{
        "source": "SQF_Food_Safety_Code.pdf",
        "clause": "2.5.5",
        "clause_title": "Internal Audits",
        "page": 34,
        "text": ("2.5.5 Internal Audits\nThe methods and responsibility for scheduling and "
                 "conducting internal audits of the SQF System shall be documented and "
                 "implemented. Internal audits shall be conducted at least annually. "
                 "Corrective action of deficiencies shall be recorded."),
    }]
    reference = ("Internal audits must be conducted at least annually. The schedule and "
                 "responsibility are documented, and corrective action for any deficiency "
                 "found is recorded.")

    print("--- Test 1: correct answer, correct citation (expect all 5s) ---")
    print(json.dumps(judge(
        "How often must internal audits be conducted?",
        reference,
        "Internal audits of the SQF System must be conducted at least annually, with the "
        "schedule and responsibility documented and corrective action for deficiencies "
        "recorded (SQF_Food_Safety_Code.pdf, 2.5.5, p.34).",
        context,
    ), indent=2))

    print("\n--- Test 2: right answer, WRONG clause (expect citation_correctness 1-2) ---")
    print(json.dumps(judge(
        "How often must internal audits be conducted?",
        reference,
        "Internal audits must be conducted at least annually "
        "(SQF_Food_Safety_Code.pdf, 2.1.4, p.12).",
        context,
    ), indent=2))

    print("\n--- Test 3: fabricated specific (expect grounding 1-2) ---")
    print(json.dumps(judge(
        "How often must internal audits be conducted?",
        reference,
        "Internal audits must be conducted at least annually, and each audit must cover a "
        "minimum of 20 percent of the SQF System elements with a maximum interval of 90 "
        "days between audits (SQF_Food_Safety_Code.pdf, 2.5.5, p.34).",
        context,
    ), indent=2))
```

### Verify the judge is sensible

```bash
uv run python evals/judge.py
```

Read the three results. Test 1 should be all 5s. Test 2 should score
`citation_correctness` 1 or 2 while `factual_correctness` stays high, which is
exactly the separation you added the axis for. Test 3 should score `grounding` 1
or 2 because the percentage and the 90-day interval appear nowhere in the
context.

If the judge does not separate those cases, tighten the rubric before running 30
questions through it. A judge you have not calibrated produces numbers that feel
like measurement and are not.

```bash
git add tracing.py evals/judge.py
git commit -m "Week 3 Day 3: tracing setup and five-axis SQF judge"
```

---

## Day 4 (Thu): The Eval Runner

### Concept: Run -> Score -> Store -> Summarize

The runner loops over the golden set and does four things: call the RAG with the
question, call the judge with the question plus reference plus answer plus
context, compute the deterministic metrics, and append a row to a CSV. Then it
prints a summary.

Probes are handled differently: there is nothing for the judge to compare, so
they skip the judge entirely and only record whether the system refused. That
keeps them cheap and keeps refusal a clean binary.

### Project: `ask.py`

The Week 2 version returns only the answer text. The judge needs the retrieved
chunks too. This complete file adds `answer_question_with_context`, keeps
`answer_question` for the CLI, adds Langfuse instrumentation (inert until Week 4),
adds `_score_answer` so every answer carries a quality signal, and adds the
strategy switch that Weeks 4 Days 10-11 will use.

`_score_answer` is the piece worth pausing on. Everything the judge produces
needs a reference answer, which only exists for the 39 golden records - so an
interactive question came back with an empty Scores tab in Langfuse, and the
system you actually use day to day was the one you had no measurements of. The
two reference-free metrics from Day 2 need nothing but the answer and the context
it was given, so they run on *every* answer, eval or not.

Note that it records `refusal` raw rather than as pass/fail. A refusal is correct
behavior on a question the corpus does not cover and a defect on one it does, and
`ask.py` has no idea which this was. `run_eval.py` does, because it holds the
golden record, and scores that judgement separately as `appropriate_refusal`.
Resisting the urge to collapse the two here is what keeps the interactive score
meaningful.

Replace `ask.py` entirely with:

```python
"""Ask a question. Retrieve context. Generate a cited answer."""
import os
import sys

from anthropic import Anthropic
from langfuse import observe, propagate_attributes

# The two metrics here are the reference-free ones: they need no golden record
# and no judge call, so they can run on any answer. The judged axes stay in the
# eval harness, which is the only place a reference answer exists.
from evals.metrics import citation_grounding, is_refusal
# Importing tracing constructs the Langfuse client from the LANGFUSE_* env vars.
# If those are unset the SDK disables itself and every @observe below is a no-op.
from tracing import langfuse

# Retrieved chunks per question. 14 rather than 5 because clause chunks are
# small (~690 chars): at k=5 the model saw ~3.4K chars and completeness was the
# weakest axis by a wide margin. A 5/10/14 sweep moved it 3.85 -> 4.09 -> 4.24
# (paired sign test p=0.013) with clause citation flat at 97%.
#
# Not higher: relevance falls monotonically over the same sweep (4.50 -> 4.29)
# as tangential excerpts dilute the answer, and context tokens scale with k on
# both the answer and judge calls. 14 is the knee, not a ceiling to raise.
TOP_K = int(os.getenv("TOP_K", "14"))
LLM_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You answer questions about the SQF certification documents in
the provided excerpts, for a food-safety practitioner.

Rules:
- Answer only from the excerpts. If they do not contain the requirement, say
  "Not found in the provided documents" and stop. Never supply a requirement
  from general knowledge or another standard.
- Cite the clause for every requirement you state, as (source, clause, p.page).
  If an excerpt has no clause number, cite the source and page.
- If an answer spans multiple clauses, list each with its own citation.
- Use plain hyphens, never em dashes."""

# vanilla is built in Week 2; hybrid and rerank arrive in Week 4 Days 10-11.
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "vanilla")

if RETRIEVAL_STRATEGY == "vanilla":
    from retrievers.vanilla import retrieve as _retrieve
elif RETRIEVAL_STRATEGY == "hybrid":
    from retrievers.hybrid import hybrid_retrieve as _retrieve
elif RETRIEVAL_STRATEGY == "rerank":
    from retrievers.rerank import rerank_retrieve as _retrieve
else:
    raise ValueError(f"Unknown RETRIEVAL_STRATEGY: {RETRIEVAL_STRATEGY}")


@observe(name="retrieve", as_type="retriever")
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Strategy-agnostic retrieval wrapper.

    Delegates to whichever retriever RETRIEVAL_STRATEGY selected and records the
    strategy, the result count and the retrieved clauses on the span, so traces
    are comparable across vanilla/hybrid/rerank runs.
    """
    chunks = _retrieve(query, k=k)
    langfuse.update_current_span(
        metadata={"strategy": RETRIEVAL_STRATEGY, "k": k, "n_results": len(chunks)},
        input={"query": query},
        output={
            "top_clauses": [c.get("clause") for c in chunks],
            "top_sources": [c.get("source") for c in chunks],
        },
    )
    return chunks


def assemble_prompt(query: str, chunks: list[dict]) -> str:
    """Build the context block. The clause and page in each header are what the
    model cites, so they must survive retrieval to get here."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[Excerpt {i} | {c['source']}"
        if c.get("clause"):
            header += f" | clause {c['clause']}"
        if c.get("page") is not None:
            header += f" | p.{c['page']}"
        header += "]"
        blocks.append(f"{header}\n{c['text']}")
    context = "\n\n---\n\n".join(blocks)
    return f"""Documentation excerpts:

{context}

---

Question: {query}

Answer using only the excerpts above, citing clauses."""


@observe(name="generate-answer", as_type="generation")
def generate(prompt: str) -> str:
    """Call Claude and log it as a Langfuse generation (model + token usage).

    Marking this as_type="generation" is what tells Langfuse to treat it as an
    LLM call: the model name and usage_details drive cost calculation and
    model-level analytics in the UI.
    """
    anthropic = Anthropic()
    response = anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.content[0].text
    langfuse.update_current_generation(
        model=LLM_MODEL,
        # Log the messages we actually sent rather than the default function
        # arg, so the system prompt is visible alongside the user content.
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        output=answer,
        model_parameters={"max_tokens": 1024},
        usage_details={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )
    return answer


def _score_answer(answer: str, chunks: list[dict]) -> None:
    """Attach the reference-free scores to the current trace.

    Runs on every answer, which is the point: the judged axes in run_eval.py
    need a reference answer, so an interactive question could never carry a
    quality signal and its Scores tab came up empty. These two need only the
    answer and the context it was given.

    Both are deliberately raw rather than pass/fail. `refusal` is correct
    behavior on a question the corpus does not cover and a defect on one it
    does, and nothing here knows which this was - run_eval.py has the golden
    record and scores that judgement separately as `appropriate_refusal`.
    """
    langfuse.score_current_trace(
        name="refusal",
        value=1 if is_refusal(answer) else 0,
        comment="1 = the answer declined to answer",
    )
    grounded = citation_grounding(answer, chunks)
    # None means the answer cited nothing, so there is nothing to be right or
    # wrong about. Recording 0.0 there would punish a clean refusal.
    if grounded is not None:
        langfuse.score_current_trace(
            name="citation_grounding",
            value=round(grounded, 3),
            comment="fraction of cited clauses that appear in the retrieved excerpts",
        )


@observe(name="rag-answer")
def answer_question_with_context(query: str) -> tuple[str, list[dict]]:
    """Run the full pipeline and return the answer AND the chunks it used.

    The eval harness needs the chunks: citation correctness and grounding can
    only be judged against the context the model actually saw, and the clause
    hit metric is computed from it directly.
    """
    chunks = retrieve(query)
    prompt = assemble_prompt(query, chunks)
    answer = generate(prompt)
    # Scored inside the span so score_current_trace has a trace to attach to.
    _score_answer(answer, chunks)
    return answer, chunks


def answer_question(query: str) -> str:
    """Answer text only. Convenience wrapper for the CLI."""
    answer, _ = answer_question_with_context(query)
    return answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python ask.py "your question"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    # Tag this trace as a CLI run so interactive questions are easy to separate
    # from eval runs in the Langfuse UI.
    with propagate_attributes(tags=["interactive", "corpus:sqf"]):
        print(answer_question(query))
    # CLI scripts exit immediately, so force-send buffered traces before we do.
    langfuse.flush()


if __name__ == "__main__":
    main()
```

Confirm nothing broke:

```bash
uv run python ask.py "How often must internal audits be conducted?"
```

### Project: `evals/run_eval.py`

Three details in this file are not obvious the first time and each one came from
a run that went wrong:

- **The refuse-to-start guard.** If `LANGFUSE_PUBLIC_KEY` is set but tracing did
  not come up, the run exits before the first API call. Absent credentials mean
  a deliberate untraced run and are fine; *broken* credentials mean someone
  meant to trace this and won't find out until they go looking for the traces.
  The CSV is only written after the loop completes, so discovering it halfway
  through costs the whole run either way. Bail on record 0, not record 20.
- **`dropped` tracking.** A record whose RAG or judge call raised is excluded
  from the summary's denominator - which means without this, a run where six
  questions errored prints the same clean-looking averages as one where none
  did, computed over a quietly smaller set. The summary now names what it lost
  before it reports anything.
- **The `cites_expected` column.** `answer_cites_expected_clause` alongside
  `hit@k`, for the reason given on Day 2: the two read different things, and
  when they disagree that disagreement is the finding.

Complete file:

```python
"""Run the golden set through the RAG, score it, save results."""
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ask import answer_question_with_context

from evals.judge import judge
from evals.metrics import (answer_cites_expected_clause, clause_hit_at_k,
                           is_refusal, retrieved_clauses)
from langfuse import propagate_attributes
from tracing import TRACING_ENABLED, langfuse

# The five judge axes, mapped to the score names they get in Langfuse.
SCORE_AXES = (
    "factual_correctness", "completeness", "relevance",
    "citation_correctness", "grounding",
)

GOLDEN_FILE = Path(__file__).resolve().parent / "golden.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Every row carries every column so the CSV is rectangular even though probe
# rows have no judge scores.
FIELDNAMES = [
    "id", "difficulty", "tags", "question", "expected_clause",
    "system_answer", "refused",
    "factual", "complete", "relevant", "citation", "grounding", "overall",
    "hit_1", "hit_3", "hit_5", "cites_expected", "top_clauses",
    "reasoning", "elapsed_sec",
]


def _blank_row(rec: dict) -> dict:
    return {
        "id": rec["id"],
        "difficulty": rec["difficulty"],
        "tags": ",".join(rec["tags"]),
        "question": rec["question"],
        "expected_clause": rec["expected_clause"] or "",
        "system_answer": "",
        "refused": "",
        "factual": "", "complete": "", "relevant": "",
        "citation": "", "grounding": "", "overall": "",
        "hit_1": "", "hit_3": "", "hit_5": "", "cites_expected": "",
        "top_clauses": "",
        "reasoning": "",
        "elapsed_sec": "",
    }


def run_eval(run_name: str, config_notes: str = "", limit: int | None = None) -> Path:
    """Run the eval, write a CSV, print a summary.

    If `limit` is set, only the first N golden records are run - handy for cheap
    smoke tests that stay under the Voyage free-tier rate limit.
    """
    # Credentials set but not working means someone meant to trace this run.
    # Bail now rather than after 39 records of API spend: the CSV is only
    # written once the loop completes, so a run discovered to be untraced
    # halfway through is a total loss either way. Absent credentials are a
    # deliberate untraced run and stay allowed.
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and not TRACING_ENABLED:
        sys.exit("Refusing to start: Langfuse credentials are set but not working, "
                 "so this run would produce no traces. See the message above.")

    records = [json.loads(line) for line in GOLDEN_FILE.open(encoding="utf-8")]
    total = len(records)
    if limit is not None:
        records = records[:limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"{timestamp}_{run_name}.csv"
    # One Langfuse session per eval run, so every question in this run shows up
    # grouped under the Sessions view and is comparable run-to-run.
    run_session = f"{run_name}-{timestamp}"
    strategy = os.getenv("RETRIEVAL_STRATEGY", "vanilla")

    rows: list[dict] = []
    # A record that errors is not in the summary's denominator, so a partial run
    # would otherwise read as a clean one. Track the losses and report them.
    dropped: dict[str, list[str]] = {"rag": [], "judge": []}
    scope = f"{len(records)} of {total}" if limit is not None else str(total)
    print(f"Running eval: {run_name} ({scope} records, strategy={strategy})")
    if config_notes:
        print(f"Config: {config_notes}")
    print()

    for i, rec in enumerate(records, 1):
        t0 = time.time()
        is_probe = rec["difficulty"] == "probe"
        # Each golden record is one trace named "eval-item". propagate_attributes
        # stamps the session id and tags onto that trace and everything nested
        # under it (the RAG pipeline + the judge call), so a run is filterable by
        # difficulty and grouped as a session.
        item_tags = ["eval", "corpus:sqf", f"strategy:{strategy}",
                     rec["difficulty"], *rec["tags"]]

        with propagate_attributes(session_id=run_session, tags=item_tags), \
                langfuse.start_as_current_observation(name="eval-item", as_type="span"):
            try:
                system_answer, chunks = answer_question_with_context(rec["question"])
            except Exception as e:
                langfuse.update_current_span(level="ERROR", status_message=f"RAG: {e}")
                print(f"  {i:2d}/{len(records)} {rec['id']} RAG ERROR: {e}")
                dropped["rag"].append(rec["id"])
                continue

            refused = is_refusal(system_answer)
            row = _blank_row(rec)
            row["system_answer"] = system_answer
            row["refused"] = refused
            row["top_clauses"] = ",".join(retrieved_clauses(chunks))
            row["elapsed_sec"] = round(time.time() - t0, 2)

            langfuse.update_current_span(
                input=rec["question"],
                output=system_answer,
                metadata={
                    "id": rec["id"],
                    "difficulty": rec["difficulty"],
                    "expected_clause": rec["expected_clause"],
                    "refused": refused,
                },
            )

            if is_probe:
                # Nothing for the judge to compare against. The only thing that
                # matters is whether the system declined, so score that directly.
                langfuse.score_current_trace(
                    name="appropriate_refusal", value=1 if refused else 0,
                    comment="probe question; expected refusal",
                )
                rows.append(row)
                mark = "REFUSED (pass)" if refused else "ANSWERED (FAIL)"
                print(f"  {i:2d}/{len(records)} {rec['id']} [probe ] {mark}")
                continue

            hits = {k: clause_hit_at_k(chunks, rec["expected_clause"], k) for k in (1, 3, 5)}
            row["hit_1"], row["hit_3"], row["hit_5"] = hits[1], hits[3], hits[5]
            # Recorded alongside hit@k because the two disagree in a way that
            # matters: hit@k reads chunk metadata, this reads the answer, so it
            # stays comparable across configurations that chunk differently.
            row["cites_expected"] = answer_cites_expected_clause(
                system_answer, rec["expected_clause"])

            try:
                scores = judge(rec["question"], rec["reference_answer"],
                               system_answer, chunks)
            except Exception as e:
                langfuse.update_current_span(level="ERROR", status_message=f"JUDGE: {e}")
                print(f"  {i:2d}/{len(records)} {rec['id']} JUDGE ERROR: {e}")
                dropped["judge"].append(rec["id"])
                rows.append(row)
                continue

            overall = sum(scores[a] for a in SCORE_AXES) / len(SCORE_AXES)
            row.update({
                "factual": scores["factual_correctness"],
                "complete": scores["completeness"],
                "relevant": scores["relevance"],
                "citation": scores["citation_correctness"],
                "grounding": scores["grounding"],
                "overall": round(overall, 3),
                "reasoning": scores["reasoning"],
            })

            # Attach the judge's verdict plus the deterministic metrics as
            # Langfuse scores - this is what turns the eval into a quality
            # signal you can filter and chart in the UI.
            for axis in SCORE_AXES:
                langfuse.score_current_trace(
                    name=axis, value=scores[axis], comment=scores["reasoning"]
                )
            langfuse.score_current_trace(name="overall", value=round(overall, 3),
                                         comment=scores["reasoning"])
            langfuse.score_current_trace(name="clause_hit_3", value=1 if hits[3] else 0,
                                         comment=f"expected {rec['expected_clause']}")

            rows.append(row)
            print(f"  {i:2d}/{len(records)} {rec['id']} [{rec['difficulty']:6s}] "
                  f"F={row['factual']} C={row['complete']} R={row['relevant']} "
                  f"Cite={row['citation']} G={row['grounding']} "
                  f"avg={overall:.2f} hit@3={'Y' if hits[3] else 'n'}")

    if not rows:
        print("No rows produced; nothing written.")
        return out_file

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written: {out_file}\n")
    print_summary(rows, dropped)

    # Force-send buffered traces and scores before the script exits.
    langfuse.flush()
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print(f"\nTraces sent to Langfuse under session '{run_session}'.")
    return out_file


def print_summary(rows: list[dict], dropped: dict[str, list[str]] | None = None) -> None:
    """Print the compliance metrics first, then the judged axes."""
    scored = [r for r in rows if r["difficulty"] != "probe" and r["overall"] != ""]
    probes = [r for r in rows if r["difficulty"] == "probe"]

    dropped = dropped or {}
    if any(dropped.values()):
        print("=== Dropped (excluded from every figure below) ===")
        for stage, ids in dropped.items():
            if ids:
                print(f"  {stage} errors: {len(ids)}  ({', '.join(ids)})")
        print()

    def avg(field, subset):
        return sum(float(r[field]) for r in subset) / len(subset)

    def pct(field, subset):
        return 100.0 * sum(1 for r in subset if r[field] is True) / len(subset)

    if probes:
        refused = sum(1 for r in probes if r["refused"] is True)
        print(f"=== Refusal probes (n={len(probes)}) ===")
        print(f"  Correctly refused:  {refused}/{len(probes)}  ({100.0*refused/len(probes):.0f}%)")

    if not scored:
        return

    false_refusals = sum(1 for r in scored if r["refused"] is True)
    print(f"\n=== Retrieval (n={len(scored)}) ===")
    print(f"  clause hit@1:       {pct('hit_1', scored):.0f}%")
    print(f"  clause hit@3:       {pct('hit_3', scored):.0f}%")
    print(f"  clause hit@5:       {pct('hit_5', scored):.0f}%")
    print(f"  cites expected:     {pct('cites_expected', scored):.0f}%")
    print(f"  False refusals:     {false_refusals}/{len(scored)}")

    print(f"\n=== Judged axes (n={len(scored)}) ===")
    print(f"  Citation:   {avg('citation', scored):.2f}")
    print(f"  Grounding:  {avg('grounding', scored):.2f}")
    print(f"  Factual:    {avg('factual', scored):.2f}")
    print(f"  Complete:   {avg('complete', scored):.2f}")
    print(f"  Relevant:   {avg('relevant', scored):.2f}")
    print(f"  Overall:    {avg('overall', scored):.2f}")

    for diff in ("easy", "medium", "hard"):
        subset = [r for r in scored if r["difficulty"] == diff]
        if subset:
            print(f"\n=== {diff.capitalize()} (n={len(subset)}) ===")
            print(f"  Overall: {avg('overall', subset):.2f}   "
                  f"hit@3: {pct('hit_3', subset):.0f}%")

    print("\n=== Lowest 5 ===")
    for r in sorted(scored, key=lambda r: float(r["overall"]))[:5]:
        print(f"  {r['id']} [{r['difficulty']:6s}] {float(r['overall']):.2f}  "
              f"{r['question'][:70]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the SQF golden set through the RAG and score it.")
    parser.add_argument("name", nargs="?", default="baseline",
                        help="Run name, used in the results filename.")
    parser.add_argument("notes", nargs="?", default="",
                        help="Free-text config notes recorded with the run.")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Only run the first N golden records (cheap smoke test).")
    args = parser.parse_args()
    run_eval(args.name, args.notes, limit=args.limit)
```

Smoke test on three records before spending a full run:

```bash
uv run python evals/run_eval.py smoke "wiring check" --limit 3
```

```bash
git add ask.py evals/run_eval.py
git commit -m "Week 3 Day 4: eval runner with clause-hit and refusal metrics"
```

---

## Day 5 (Fri): First Full Run + Analysis

### Run the baseline

```bash
TOP_K=5 uv run python evals/run_eval.py baseline "vanilla cosine, top_k=5, clause chunking"
```

39 records: 39 RAG calls plus 34 judge calls, sequential. Budget about 15
minutes - the wall clock is dominated by the Voyage free tier's 3 requests/minute
on the query embedding, not by Claude.

Output looks like this. **The numbers below are illustrative** - a deliberately
mediocre first run, not this project's measured results. Your own first run is
the only baseline that means anything, and `EVAL_REPORT.md` is where the real
figures live.

```
Running eval: baseline (39 records, strategy=vanilla)
Config: vanilla cosine, top_k=5, clause chunking

   1/39 sys-elements-e1     [easy  ] F=5 C=4 R=5 Cite=5 G=5 avg=4.80 hit@3=Y
   2/39 sys-elements-m1     [medium] F=4 C=3 R=5 Cite=3 G=4 avg=3.80 hit@3=Y
   3/39 sys-elements-h1     [hard  ] F=3 C=2 R=4 Cite=2 G=3 avg=2.80 hit@3=n
   ...
  35/39 probe-finished-product-micro-limit [probe ] REFUSED (pass)
  36/39 probe-iso-22000-team-leader        [probe ] ANSWERED (FAIL)
  ...

=== Refusal probes (n=5) ===
  Correctly refused:  4/5  (80%)

=== Retrieval (n=34) ===
  clause hit@1:       43%
  clause hit@3:       67%
  clause hit@5:       73%
  cites expected:     71%
  False refusals:     2/34

=== Judged axes (n=34) ===
  Citation:   3.53
  Grounding:  4.07
  Factual:    3.87
  Complete:   3.43
  Relevant:   4.20
  Overall:    3.82
```

That is your baseline. Read it in the order the summary prints it, which is
deliberate: anything dropped first, then refusal behavior, then retrieval, then
answer quality. A `Dropped` block above the probes means those records errored
and are in none of the figures below it - fix that before reading anything else,
because every average printed under it was computed over a smaller set than you
think.

### Read the failures

Open the CSV and look at three things.

**Probe failures.** Any probe that got an answer instead of a refusal is your
most serious defect. Read `top_clauses` for that row: retrieval almost certainly
surfaced something adjacent and the model treated it as close enough. That is the
fabrication failure mode, caught.

**hit@3 misses on scored questions.** Compare `expected_clause` against
`top_clauses`. If the right clause never appeared, this is a retrieval failure and
Week 4 is aimed squarely at it. If the right clause did appear but the answer
still scored low, it is a generation failure and the fix is the prompt.

**Citation scores below 3 where factual is 4-5.** These are the cases the extra
axis exists to catch: right answer, wrong clause. Read the reasoning column to see
what the judge saw.

Write your observations in `NOTES.md` as you go. You will need them Sunday.

### Project: `evals/analyze.py`

Complete file:

```python
"""Analysis of SQF eval result CSVs.

Usage:
  uv run python evals/analyze.py                              # summarize latest
  uv run python evals/analyze.py RESULTS.csv                  # summarize one
  uv run python evals/analyze.py BASE.csv VARIANT.csv         # compare two
  uv run python evals/analyze.py compare_three V.csv H.csv R.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.metrics import answer_cites_expected_clause, is_refusal

AXES = ["citation", "grounding", "factual", "complete", "relevant", "overall"]
HITS = ["hit_1", "hit_3", "hit_5"]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _latest_results() -> Path:
    """Newest results CSV, or exit with a usable message if there are none."""
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No result CSVs in {RESULTS_DIR}. "
                 f"Run 'uv run python evals/run_eval.py' first, "
                 f"or pass a CSV path explicitly.")
    return csvs[-1]


def _split(csv_path: str):
    """Return (scored, probes). Probe rows have no judge scores."""
    df = pd.read_csv(csv_path)
    for col in HITS + ["refused"]:
        if col in df.columns:
            df[col] = df[col].astype("object")
    probes = df[df["difficulty"] == "probe"]
    scored = df[(df["difficulty"] != "probe") & df["overall"].notna()]
    return scored, probes


def _hit_rate(df: pd.DataFrame, col: str) -> float:
    vals = df[col].map(lambda v: str(v).lower() == "true")
    return 100.0 * vals.mean() if len(df) else float("nan")


def _refusal_rate(df: pd.DataFrame) -> float:
    """Recomputed from the answer text, not read from the stored column.

    Both derived metrics are pure functions of text the CSV already holds, so
    recomputing costs nothing and means a run scored under an older, buggier
    metric reports correctly without being re-run. That matters here: the
    refusal regex was widened after five runs had already been scored, and the
    `refused` column in those files understates the true rate by up to 40 points.
    """
    if not len(df):
        return float("nan")
    return 100.0 * df["system_answer"].map(lambda a: is_refusal(str(a))).mean()


def _cites_rate(df: pd.DataFrame) -> float:
    """Share of answers citing an expected clause. See metrics.py for why this
    is the metric to compare across different chunking strategies."""
    if not len(df):
        return float("nan")
    vals = df.apply(
        lambda r: answer_cites_expected_clause(str(r["system_answer"]),
                                               str(r["expected_clause"])),
        axis=1)
    return 100.0 * vals.mean()


def summarize(csv_path: str) -> None:
    scored, probes = _split(csv_path)

    print(f"\n=== {csv_path} ===")
    print(f"Scored: {len(scored)}   Probes: {len(probes)}\n")

    if len(probes):
        print(f"Probe refusal rate:  {_refusal_rate(probes):.0f}%  "
              f"(higher is better)")
    if not len(scored):
        return
    print(f"False refusal rate:  {_refusal_rate(scored):.0f}%  (lower is better)")

    print("\nClause hit rate:")
    for col in HITS:
        print(f"  {col}: {_hit_rate(scored, col):5.1f}%")
    print(f"  cites expected: {_cites_rate(scored):5.1f}%  "
          f"(comparable across chunking strategies)")

    print("\nAverage by axis:")
    print(scored[AXES].mean().round(2).to_string())

    print("\nAverage by difficulty:")
    print(scored.groupby("difficulty")[AXES].mean().round(2).to_string())

    # Tag analysis: which requirement areas does the system handle worst?
    tag_rows = []
    for _, row in scored.iterrows():
        for tag in str(row["tags"]).split(","):
            if tag.strip():
                tag_rows.append({"tag": tag.strip(), "overall": row["overall"],
                                 "citation": row["citation"]})
    if tag_rows:
        tag_df = pd.DataFrame(tag_rows)
        print("\nWorst 10 tags by overall:")
        print(tag_df.groupby("tag")[["overall", "citation"]]
              .agg(["mean", "count"]).round(2)
              .sort_values(("overall", "mean")).head(10).to_string())


def compare(baseline_csv: str, variant_csv: str) -> None:
    a, a_probes = _split(baseline_csv)
    b, b_probes = _split(variant_csv)

    print(f"Baseline: {baseline_csv}")
    print(f"Variant:  {variant_csv}\n")

    print("Compliance metrics:")
    print(f"  probe refusal   baseline={_refusal_rate(a_probes):5.1f}%  "
          f"variant={_refusal_rate(b_probes):5.1f}%")
    print(f"  false refusal   baseline={_refusal_rate(a):5.1f}%  "
          f"variant={_refusal_rate(b):5.1f}%")
    for col in HITS:
        av, bv = _hit_rate(a, col), _hit_rate(b, col)
        print(f"  {col:13s} baseline={av:5.1f}%  variant={bv:5.1f}%  "
              f"delta={bv-av:+.1f}")
    av, bv = _cites_rate(a), _cites_rate(b)
    print(f"  {'cites expected':13s} baseline={av:5.1f}%  variant={bv:5.1f}%  "
          f"delta={bv-av:+.1f}")

    print("\nJudged axes (variant - baseline):")
    for col in AXES:
        delta = b[col].mean() - a[col].mean()
        print(f"  {col:10s} baseline={a[col].mean():.2f}  "
              f"variant={b[col].mean():.2f}  delta={delta:+.2f}")

    merged = a.merge(b, on="id", suffixes=("_a", "_b"))
    merged["delta"] = merged["overall_b"] - merged["overall_a"]
    print("\nBiggest improvements:")
    for _, row in merged.nlargest(3, "delta").iterrows():
        print(f"  {row['id']} {row['overall_a']:.2f} -> {row['overall_b']:.2f}  "
              f"{str(row['question_a'])[:65]}")
    print("\nBiggest regressions:")
    for _, row in merged.nsmallest(3, "delta").iterrows():
        print(f"  {row['id']} {row['overall_a']:.2f} -> {row['overall_b']:.2f}  "
              f"{str(row['question_a'])[:65]}")


def compare_three(vanilla_csv: str, hybrid_csv: str, rerank_csv: str) -> None:
    v, vp = _split(vanilla_csv)
    h, hp = _split(hybrid_csv)
    r, rp = _split(rerank_csv)

    print(f"{'Metric':16s}  {'Vanilla':>9s}  {'Hybrid':>9s}  {'Rerank':>9s}  "
          f"{'H-V':>7s}  {'R-V':>7s}")
    print("-" * 68)

    # Compliance metrics first - these are the headline for this corpus.
    for label, frames in (("probe refusal %", (vp, hp, rp)),):
        vm, hm, rm = (_refusal_rate(f) for f in frames)
        print(f"{label:16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
              f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")
    vm, hm, rm = (_refusal_rate(f) for f in (v, h, r))
    print(f"{'false refusal %':16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
          f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")
    for col in HITS:
        vm, hm, rm = (_hit_rate(f, col) for f in (v, h, r))
        print(f"{col + ' %':16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
              f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")
    vm, hm, rm = (_cites_rate(f) for f in (v, h, r))
    print(f"{'cites expected %':16s}  {vm:>9.1f}  {hm:>9.1f}  {rm:>9.1f}  "
          f"{hm-vm:>+7.1f}  {rm-vm:>+7.1f}")

    print()
    for col in AXES:
        vm, hm, rm = v[col].mean(), h[col].mean(), r[col].mean()
        print(f"{col:16s}  {vm:>9.2f}  {hm:>9.2f}  {rm:>9.2f}  "
              f"{hm-vm:>+7.2f}  {rm-vm:>+7.2f}")

    for diff in ("easy", "medium", "hard"):
        vs = v[v["difficulty"] == diff]["overall"].mean()
        hs = h[h["difficulty"] == diff]["overall"].mean()
        rs = r[r["difficulty"] == diff]["overall"].mean()
        print(f"\n{diff:8s}  vanilla={vs:.2f}  hybrid={hs:.2f}  rerank={rs:.2f}")

    merged = v.merge(r, on="id", suffixes=("_v", "_r"))
    merged["delta"] = merged["overall_r"] - merged["overall_v"]
    print("\nBiggest improvements (rerank vs vanilla):")
    for _, row in merged.nlargest(5, "delta").iterrows():
        print(f"  {row['id']} {row['overall_v']:.2f} -> {row['overall_r']:.2f}  "
              f"{str(row['question_v'])[:65]}")
    print("\nBiggest regressions (rerank vs vanilla):")
    for _, row in merged.nsmallest(5, "delta").iterrows():
        print(f"  {row['id']} {row['overall_v']:.2f} -> {row['overall_r']:.2f}  "
              f"{str(row['question_v'])[:65]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "compare_three":
        if len(args) != 4:
            sys.exit("usage: analyze.py compare_three VANILLA.csv HYBRID.csv RERANK.csv")
        compare_three(args[1], args[2], args[3])
    elif len(args) >= 2:
        compare(args[0], args[1])
    elif len(args) == 1:
        summarize(args[0])
    else:
        summarize(str(_latest_results()))
```

Two things in here are worth copying regardless of corpus. `_refusal_rate` and
`_cites_rate` are **recomputed from `system_answer`** rather than read out of the
stored columns. Both are pure functions of text the CSV already holds, so it
costs nothing - and it means a run scored under an older, buggier metric reports
correctly without being re-run. That is not hypothetical: the refusal regex was
widened after five runs had already been scored, and the `refused` column in
those files understates the true rate by up to 40 points. Recomputation turned a
re-run-everything problem into a no-op.

```bash
uv run python evals/analyze.py
git add evals/analyze.py
git commit -m "Week 3 Day 5: baseline run and analysis tooling"
```

---

## Day 6 (Sat): The Experimental Variant

The day the eval earns its keep. Change one thing. Re-run. Observe the delta.

### Pick one experiment

Do not change three things at once; the point is to attribute the delta to a
specific change. Three good candidates for this corpus:

**Option A: clause-aware chunking vs a fixed window.** Temporarily set your
chunker to a 2000-character sliding window ignoring clause boundaries, reindex,
re-run. This is the highest-information experiment available to you, because it
puts a number on the central design decision of Week 2. Expect citation and
hit@3 to fall noticeably while relevance barely moves, which is exactly the
argument for clause chunking.

**Option B: `TOP_K` 5 to 10.** No reindex needed. More context may help synthesis
questions. Watch the probe refusal rate: more context is exactly what tempts a
model to answer something it should refuse, and if refusal drops this is a
regression even if `overall` rises.

**Option C: drop the refusal instruction from `SYSTEM_PROMPT`.** Re-run and watch
probe refusal collapse. A destructive experiment that produces a number you can
put in the report: "removing the refusal instruction dropped probe refusal from
80% to 20% while overall answer quality changed by less than 0.1."

Pick one. A or C give the most informative result.

### Run and compare

```bash
uv run python evals/run_eval.py variant_chunking "fixed 2000-char window, no clause split"
uv run python evals/analyze.py evals/results/TIMESTAMP_baseline.csv evals/results/TIMESTAMP_variant_chunking.csv
```

### Do not tune to your eval

The trap: keep tweaking until the average goes up, ship, declare victory. Thirty
questions is a small sample. Make 20 changes and pick the best and you have
overfit to your own measuring instrument. Run 2-3 informed variants and pick the
one with a clear causal story, not the highest number.

```bash
# Restore your config to whichever variant you decided is best, or leave at
# baseline and document the choice in EVAL_REPORT.md
git add -A
git commit -m "Week 3 Day 6: variant experiment"
```

---

## Day 7 (Sun): EVAL_REPORT.md + Polish

### Create `EVAL_REPORT.md`

Template to fill with your real numbers:

```markdown
# Evaluation Report: cert-rag-cli (SQF certification corpus)

## Method

A 30-question golden set over our SQF certification documents, plus 5
out-of-corpus probe questions where the correct behavior is refusal. Scored
questions span easy (single clause lookup), medium (synthesis across 2-3
clauses), and hard (edge cases, comparisons, troubleshooting), tagged by
requirement area.

Two kinds of measurement:

1. Deterministic, no LLM call. `clause_hit@k` asks whether the clause that
   actually contains the requirement was retrieved in the top k. `refusal rate`
   asks whether the system declined on the probe set. Both are stable across
   runs and immune to judge drift.
2. LLM-as-judge. Claude Sonnet 4.6 scores five axes 1-5 with brief reasoning,
   given the question, the reference answer, and the excerpts the system
   actually retrieved: factual correctness, completeness, relevance, citation
   correctness, and grounding.

Citation correctness and grounding exist because in a compliance setting a
fluent answer citing the wrong clause is worse than a refusal. The three
standard axes do not catch that failure.

## Corpus

<N> SQF documents (PDF and DOCX, <M> of them scanned and OCRed), chunked on
clause boundaries rather than a fixed window, yielding <K> chunks indexed in
Chroma with clause, page, module and doc_type metadata. Clause attribution:
<n> chunks from a header in the text, <n> from the filename, <n> with no clause.

## Baseline configuration

- Chunking: clause-aware, sub-split above 2400 characters
- Embedding: Voyage voyage-3-lite
- Retrieval: cosine similarity, top_k=5
- Generation: Claude Sonnet 4.6, refusal-first system prompt

## Baseline results

Compliance metrics:

| Metric | Value |
|---|---|
| Probe refusal rate | <n>/<5> |
| False refusal rate (scored) | <n>/34 |
| clause hit@1 | <n>% |
| clause hit@3 | <n>% |
| clause hit@5 | <n>% |
| Answer cites expected clause | <n>% |

Judged axes:

| Axis | Score | Notes |
|---|---|---|
| Citation correctness | | |
| Grounding | | |
| Factual correctness | | |
| Completeness | | |
| Relevance | | |
| Overall | | |

By difficulty:

| Difficulty | Overall | hit@3 |
|---|---|---|
| Easy | | |
| Medium | | |
| Hard | | |

Worst requirement areas by tag:

| Tag | Overall | Citation | Count |
|---|---|---|---|

## Experimental variant: <what you changed>

Delta vs baseline:

- clause hit@3: <+/-n>%
- Probe refusal: <+/-n>
- Citation: <+/-n.nn>
- Grounding: <+/-n.nn>
- Overall: <+/-n.nn>

Three observations:
1.
2.
3.

## What this measured and what it didn't

This measures whether the system answers a small hand-curated set of SQF
questions usefully and cites the right clause, as judged by a strong LLM plus
deterministic clause matching. It does not measure: latency, cost, robustness
to adversarial phrasing, performance on documents outside the indexed set,
or whether the reference answers themselves are correct readings of the code.
The reference answers were written by one person from the documents and have
not been reviewed by a second practitioner. That is the largest single threat
to the validity of these numbers.

## What I'd do next

1. Second-reader review of the 34 reference answers and expected clauses
2. Larger golden set (100+) for more stable averages, and more than 5 probes
3. Multiple runs per config to estimate judge variance
4. Hybrid retrieval and reranking (Week 4)
5. Separate the retrieval metric from the generation metric fully, so a failure
   is attributable without reading the CSV
```

Fill in real numbers. Keep it honest. The "what this didn't measure" section is
what distinguishes a real engineer's eval report from theater, and the note about
unreviewed reference answers is the most credible sentence in the document.

Two additions this project's finished `EVAL_REPORT.md` grew that are worth
planning for. First, **a judge-variance section**: two runs of the identical
configuration against the identical index, a day apart, to establish the noise
floor before interpreting any delta. Without it you have no basis for calling a
0.05 movement a result. Second, **a "defect found and fixed" section**. You will
find a metric bug partway through, and the honest move is to say what it was,
what it cost, and what pins it now - not to quietly re-run and report the clean
numbers.

### Week 3 Wrap-up Checklist

- [ ] Scored questions and 5 probes in `evals/golden.jsonl`, all with real clause numbers from your documents, covering every requirement area
- [ ] `validate.py` passes (schema, refusal convention, clause grounding, coverage)
- [ ] `check_metrics.py` passes
- [ ] Judge calibration tests separate correct / wrong-clause / fabricated cases
- [ ] Baseline run complete, CSV in `evals/results/`
- [ ] One variant run and compared
- [ ] `EVAL_REPORT.md` written with real numbers
- [ ] You can explain why citation correctness and grounding are separate axes

---

# WEEK 4: Observability + Advanced Retrieval

---

## Day 8 (Mon): Langfuse

### Concept: What Tracing Is

If you have used New Relic or Datadog, tracing for LLMs is the same idea: every
operation is recorded with inputs, outputs, latency and metadata, viewable in a
UI. For LLM systems specifically, traces capture the full prompt including system
prompt and context, the full response, token counts, latency and cost per call,
any tool calls, and a hierarchical parent/child structure so the retrieval call
shows as a child of the answer operation.

Why it matters in production: when someone says "the answer it gave me last
Tuesday was wrong," you find that session, see exactly which chunks were
retrieved and what prompt was sent, and debug. Without tracing you are guessing.

For an interview, knowing tracing exists separates "I prototyped" from "I
shipped." One tool known deeply is enough.

### Concept: Self-hosted vs SaaS

Langfuse offers cloud and self-hosted. Self-host locally via Docker Compose for
learning. The hosted version is fine, but running it yourself demonstrates you
can operate it, which is what a customer actually asks.

### Project: Stand up Langfuse

```bash
cd ~/projects
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Wait 60 seconds, then visit `http://localhost:3000`. Create an account
(local-only, no email verification). Create a project called `cert-rag`. Go to
Settings -> API Keys -> Create new API keys. Add them to the project's `.env`
(the file `env.py` loads at import - no `source` step, a fresh terminal just
works):

```bash
# append to .env (gitignored)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

`tracing.py` now checks these once at startup. If a key is wrong you get one
clear "TRACING DISABLED" banner instead of a silent per-span 401, and
`run_eval.py` refuses to start a traced run it cannot actually trace. Confirm
they took:

```bash
cd ~/projects/cert-rag-cli
uv run python -c "import tracing; print('tracing enabled:', tracing.TRACING_ENABLED)"
```

### The payoff: no code change

Your code has been instrumented since Week 3 Day 3. The decorators were no-ops
because the keys were absent. Now they are not:

```bash
cd ~/projects/cert-rag-cli
uv run python ask.py "How often must internal audits be conducted?"
```

Refresh the Langfuse UI. You should see a trace named `rag-answer` with two
children: `retrieve` (showing the retrieved clauses in its output) and
`generate-answer` (showing the model, the full prompt including system prompt,
and token usage with computed cost).

If you see nothing, the usual cause is `LANGFUSE_HOST` not being read. Print your
env vars and check. If you set up a separate Langfuse project per corpus you can
skip the `corpus:sqf` tag, but leaving it costs nothing and makes filtering
unambiguous if you ever point both projects at one instance.

### Run the eval with tracing on

```bash
uv run python evals/run_eval.py traced_baseline "vanilla, tracing on"
```

In the UI, open Sessions. You will see one session per run, with 35 traces under
it. Each `eval-item` trace carries the judge's five scores plus `clause_hit_3`
attached as Langfuse scores, and is tagged with difficulty, strategy and the
question's requirement tags. Filter by `citation_correctness < 3` and you have a
one-click list of every question where the system cited the wrong clause, with
the retrieved chunks right there.

That is the workflow the instrumentation exists for.

### Reading (45 min)

- Langfuse quickstart: https://langfuse.com/docs/get-started
- Langfuse Python decorators: https://langfuse.com/docs/sdk/python/decorators
- Langfuse scores: https://langfuse.com/docs/scores

No code commit today; the Langfuse install lives outside your repo.

---

## Day 9 (Tue): Instrument the Retrievers

Right now `retrieve` in `ask.py` is one span. The embedding call inside it is
invisible, which means when a query is slow you cannot tell whether it was Voyage
or Chroma. Push instrumentation down into the retriever layer, and while you are
here, add the Voyage pacing the free tier needs during eval runs.

Two guards, because they cover different failures. **Pacing** never issues calls
faster than the tier allows in the first place; **retry** recovers anyway when a
limit is hit despite pacing (another process sharing the key, a transient server
error). Pacing is the one that carries a full eval: a 39-record run paces to
about 2.6 calls/min and finishes in ~15 minutes with no rejected call. Drop it
and the run bursts straight past the limit, then every call pays an
unpredictable backoff instead of a predictable interval. Both endpoints (embed
and rerank) bill against the same account limit, so the pacing gate is shared and
lives in one place, `paced_call`.

### Project: `retrievers/embed.py`

Complete file, replacing the Week 2 version:

```python
"""Query embedding via Voyage, with rate-limit pacing and backoff.

Lives in the retrievers package (not ask.py) so every retrieval strategy can
share it without importing ask.py, which would be a circular import - ask.py
imports a retriever at module load.

Voyage's free tier allows 3 requests per minute. Two guards below, because they
cover different failures:

  pacing  - never issue calls faster than the tier allows in the first place
  retry   - recover anyway when a limit is hit, since pacing cannot account for
            other processes sharing the same API key

Pacing is the one that carries a full eval. A 39-record run paces to about 2.6
calls/min and finishes in ~15 minutes with no rejected call. Drop it and the
run bursts straight past the limit, then every call pays an unpredictable
backoff instead of a predictable interval.

Set VOYAGE_MIN_INTERVAL_SEC=0 to disable pacing once the account has a payment
method and standard rate limits.
"""
import os
import sys
import time

import voyageai
from langfuse import observe
from voyageai import error as voyage_error

from tracing import langfuse

EMBED_MODEL = "voyage-3-lite"

# 3 RPM means one call every 20s; 21 leaves a margin for clock skew, matching
# the SLEEP_BETWEEN_BATCHES constant embed.py uses on the ingest side.
MIN_INTERVAL_SEC = float(os.getenv("VOYAGE_MIN_INTERVAL_SEC", "21"))

# Transient by nature: waiting and retrying is the correct response. Auth and
# malformed-request errors are deliberately absent - retrying those just turns a
# clear failure into a slow one. Matching on the exception type rather than on
# substrings of its message, so a server error is not misread as throttling
# because its text happened to contain a number.
RETRYABLE = (
    voyage_error.RateLimitError,
    voyage_error.ServerError,
    voyage_error.ServiceUnavailableError,
    voyage_error.APIConnectionError,
)

# One client for the process. Constructing one per call re-read the environment
# and discarded any connection reuse for no benefit.
_client = None
# None rather than 0.0 means "no call yet". monotonic()'s zero point is
# undefined, so a real reading can legitimately be 0.0 and a truthiness test
# would silently skip the first interval.
_last_call_at: float | None = None


def _voyage() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client()
    return _client


def _wait_for_slot() -> None:
    """Sleep until MIN_INTERVAL_SEC has passed since the previous call."""
    global _last_call_at
    if MIN_INTERVAL_SEC > 0 and _last_call_at is not None:
        elapsed = time.monotonic() - _last_call_at
        if elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)
    _last_call_at = time.monotonic()


def paced_call(operation: str, *, max_retries: int = 6, **kwargs):
    """Run one Voyage API call under the process-wide pace and retry policy.

    Every Voyage endpoint bills against the same account rate limit, so they all
    queue behind this one gate. That matters for the rerank strategy, which
    spends two requests per question - one embedding, one rerank. Pacing only
    the embedding would let the rerank half slip past the limit unmetered and
    put the run straight back into the 429s pacing exists to avoid.

    `operation` names the method on the client: "embed", "rerank".

    Retries back off exponentially (10s, 20s, 40s, then capped at 60s) so a call
    that arrives mid-window waits out the whole per-minute bucket rather than
    hammering it. Pacing still applies between attempts, so the two compose to
    max(interval, backoff): the first couple of retries land on the 21s pacing
    floor and the later ones dominate it.

    Raises the last error if every attempt fails, so a genuinely dead API still
    surfaces as a RAG error in the eval rather than being silently swallowed.
    """
    for attempt in range(max_retries + 1):
        _wait_for_slot()
        try:
            return getattr(_voyage(), operation)(**kwargs)
        except RETRYABLE as err:
            if attempt == max_retries:
                raise
            delay = min(60, 10 * (2 ** attempt))
            print(f"    {type(err).__name__} from Voyage {operation}, retrying in "
                  f"{delay}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(delay)


@observe(name="embed-query", as_type="embedding",
         capture_input=False, capture_output=False)
def embed_query(query: str, max_retries: int = 6) -> list[float]:
    """Embed a query, paced and retried by paced_call.

    Traced as an "embedding" observation. We suppress the default input/output
    capture and set them by hand: logging the query text is useful, but the raw
    float vector is noise in the UI, so we record only its dimensionality.
    """
    langfuse.update_current_generation(model=EMBED_MODEL, input=query)
    response = paced_call("embed", max_retries=max_retries,
                          texts=[query], model=EMBED_MODEL, input_type="query")
    embedding = response.embeddings[0]
    langfuse.update_current_generation(metadata={"dimensions": len(embedding)})
    return embedding
```

### Project: `retrievers/vanilla.py`

Complete file, replacing the Week 2 version:

```python
"""Vanilla cosine-similarity retrieval."""
import chromadb
from langfuse import observe

from retrievers.embed import embed_query

CHROMA_DIR = ".chroma"
COLLECTION_NAME = "sqf_docs"


@observe(name="vector-search", as_type="retriever")
def retrieve(query: str, k: int = 5) -> list[dict]:
    """Embed the query and pull the k nearest chunks from Chroma.

    Traced as a "retriever" observation - the embedding call nests underneath
    it, so a slow query is attributable to Voyage or Chroma at a glance.

    The clause, clause_title and page fields are carried through from Chroma
    metadata. They are what ask.py puts in the excerpt headers for the model to
    cite, and what evals/metrics.py matches against expected_clause. Drop them
    here and citation breaks everywhere downstream.
    """
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION_NAME)

    q_emb = embed_query(query)  # note: input_type='query', not 'document'
    results = collection.query(query_embeddings=[q_emb], n_results=k)

    return [
        {
            "text": doc,
            "source": meta["source"],
            "clause": meta.get("clause"),
            "clause_title": meta.get("clause_title"),
            "page": meta.get("page"),
            "doc_type": meta.get("doc_type"),
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
```

```bash
uv run python ask.py "What must the food defense plan contain?"
```

Check the Langfuse UI: `rag-answer` -> `retrieve` -> `vector-search` ->
`embed-query`, four levels deep, with the embedding call timed separately.

```bash
git add retrievers/embed.py retrievers/vanilla.py
git commit -m "Week 4 Day 9: instrument retrievers, add Voyage rate-limit backoff"
```

---

## Day 10 (Wed): BM25 + Hybrid Retrieval

### Concept: Why hybrid

Vector search finds chunks that are semantically similar to the query. It is
excellent at paraphrase: "how often do we check our own system" matches a clause
about internal audits even with no word overlap.

Its weakness is rare terms and exact identifiers. Search for clause `2.5.5` and
the embedding for that string sits near the embedding for any clause number. The
vector search happily returns 2.5.3, 2.5.4 and 2.6.1 because they are all "clause
number things." A keyword search returns only chunks that literally contain
`2.5.5`.

This matters more for a compliance corpus than for prose documentation. A large
share of real queries are lexical and precise: a clause number, "allergen," a
verbatim requirement phrase someone is checking. And SQF terminology is
unforgiving in the same way code identifiers are: "corrective action" and
"preventative action" are different requirements, "verification" and
"validation" are different requirements, and an embedder will place them close
together because they are conceptually adjacent.

The fix is hybrid: run both, fuse the results. For most production RAG systems
hybrid is the default. Pure vector is for prototypes.

### Concept: BM25

The standard keyword-search algorithm. TF-IDF with bonuses for term saturation
and document-length normalization. You do not need the math; you need to know it
scores documents by how well their words match the query, weighted by rarity.

### Concept: Reciprocal Rank Fusion

The simplest way to combine two retrievers: for each document, sum
`1 / (rank + k)` across the lists it appears in. A document ranked 1st in vector
and 5th in BM25 scores about `1/61 + 1/65`. One ranked 1st in only one list
scores `1/61`. The constant `k` (typically 60) stops the top result dominating.
RRF is robust to score-scale differences, which matters because cosine distances
and BM25 scores live in completely different ranges.

### Concept: the tokenizer is the whole game here

A standard tokenizer splits on every non-alphanumeric character. Run `2.5.5`
through `re.findall(r"\w+", text)` and you get `["2", "5", "5"]`. The clause
number, the single most precise query signal in your corpus, is destroyed before
BM25 ever sees it, and the tokens that survive are so common they carry no
information.

Preserving dotted alphanumerics is a two-line change and it is the difference
between BM25 helping and BM25 being noise. This is the kind of corpus-specific
engineering decision worth naming in an interview.

### Project: `retrievers/hybrid.py`

Complete file:

```python
"""Hybrid retrieval: BM25 keyword + cosine vector, fused via RRF."""
import json
import re
from pathlib import Path

from langfuse import observe
from rank_bm25 import BM25Okapi

from retrievers.vanilla import retrieve as vector_retrieve
from tracing import langfuse

# Anchored to the repo root rather than the cwd. Evals happen to run from the
# repo root, but serve.py can be launched from anywhere, and a relative path
# would make BM25 fail depending only on where the process was started.
CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"

# Built once at first use and cached for the process.
_chunks_cache: list[dict] | None = None
_bm25_cache: BM25Okapi | None = None

# Keep dotted alphanumerics intact so clause numbers survive tokenization:
#   "clause 2.5.5 internal audits" -> ["clause", "2.5.5", "internal", "audits"]
# A plain \w+ pattern would yield ["clause", "2", "5", "5", ...], destroying the
# most precise lexical signal this corpus has.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)+|[a-z0-9]+")

# Indexing clause-number parents alongside each token ("2.1.3.5" also emitting
# "2.1.3" and "2.1") was tried here and reverted. It does what it claims - the
# query token "2.1" goes from reaching 12 chunks to 61, and the one golden
# question naming a bare clause improves from rank 14 to 6 - but it lowers BM25
# clause hit@14 across the scored set from 97.1% to 94.1% and leaves hit@28
# unchanged. The three questions it pushes out of the window contain no clause
# number at all: they lose because parent tokens let broad section-level chunks
# outrank specific sub-clauses. That is the same failure hybrid already has (see
# EVAL_REPORT Experiment 3), so the expansion makes the real problem worse while
# fixing a cosmetic one. Only 3 of 39 golden questions name a clause at all.


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _index_text(chunk: dict) -> str:
    """What BM25 indexes for a chunk.

    Includes the clause number and title alongside the body so a query like
    "internal audits" matches the clause heading even when the body phrases the
    requirement differently.
    """
    parts = [chunk.get("clause") or "", chunk.get("clause_title") or "", chunk["text"]]
    return " ".join(p for p in parts if p)


def _load_bm25() -> tuple[list[dict], BM25Okapi]:
    global _chunks_cache, _bm25_cache
    if _bm25_cache is None:
        if not CHUNKS_FILE.exists():
            # data/ is gitignored, so a fresh clone has the code but not the
            # corpus. Say which step is missing instead of a bare IO error.
            raise FileNotFoundError(
                f"{CHUNKS_FILE} not found. Hybrid retrieval indexes the same chunks "
                "the vector store was built from - run ingest.py, then chunk.py, "
                "then embed.py."
            )
        _chunks_cache = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
        _bm25_cache = BM25Okapi([_tokenize(_index_text(c)) for c in _chunks_cache])
    return _chunks_cache, _bm25_cache


def _key(result: dict) -> tuple:
    """Identity used to fuse the two result lists.

    The retrievers read different stores - vector from Chroma, BM25 from
    chunks.jsonl - and share no id, because vanilla.retrieve does not return
    one. Chroma holds the chunk text verbatim, so text plus provenance is a
    stable join key across both.

    Deliberately the whole text, not a prefix: two chunks in this corpus share
    their first 100 characters ("INTRODUCTION\\nSpices, Inc. has established,
    documented, and implemented procedure..."), so a prefix key fuses two
    distinct chunks into one and silently drops a result.
    """
    return (result.get("source"), result.get("page"), result["text"])


@observe(name="bm25-search", as_type="retriever")
def bm25_retrieve(query: str, k: int = 10) -> list[dict]:
    chunks, bm25 = _load_bm25()
    scores = bm25.get_scores(_tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = [
        {
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "clause": chunks[i].get("clause"),
            "clause_title": chunks[i].get("clause_title"),
            "page": chunks[i].get("page"),
            "doc_type": chunks[i].get("doc_type"),
            "rank": rank + 1,
            "score": float(scores[i]),
        }
        for rank, i in enumerate(top)
    ]
    langfuse.update_current_span(
        input={"query": query, "tokens": _tokenize(query)},
        output={"top_clauses": [r["clause"] for r in results]},
    )
    return results


@observe(name="hybrid-search", as_type="retriever")
def hybrid_retrieve(query: str, k: int = 5, k_per_retriever: int | None = None,
                    rrf_k: int = 60) -> list[dict]:
    """Run vector + BM25, fuse via Reciprocal Rank Fusion.

    k_per_retriever defaults to max(2k, 20) rather than a fixed 10. ask.py asks
    for TOP_K=14, and a fixed 10 would hand the fusion fewer vector candidates
    than the vanilla retriever sees at that same k - hybrid would then be
    measured against a baseline it was never given the depth to match. Depth is
    nearly free here: both retrievers run off the one embedding call, and BM25
    scores the whole corpus regardless of k.
    """
    if k_per_retriever is None:
        k_per_retriever = max(2 * k, 20)

    vector_results = vector_retrieve(query, k=k_per_retriever)
    bm25_results = bm25_retrieve(query, k=k_per_retriever)

    rrf_scores: dict[tuple, float] = {}
    seen: dict[tuple, dict] = {}

    # Derive rank from list position (1-based). The vector retriever returns
    # results nearest-first but with no "rank" key, so don't rely on one.
    for rank, r in enumerate(vector_results, 1):
        key = _key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rank + rrf_k)
        seen[key] = r

    for rank, r in enumerate(bm25_results, 1):
        key = _key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rank + rrf_k)
        # Keep the vector copy when both retrievers surfaced the chunk: it
        # carries the cosine distance, which the BM25 record has no analogue for.
        seen.setdefault(key, r)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    # "score" is the fused RRF score, replacing whatever per-retriever score the
    # source record carried. "rank" is the position after fusion.
    results = [
        {**seen[key], "score": score, "rank": rank + 1}
        for rank, (key, score) in enumerate(ranked)
    ]
    langfuse.update_current_span(
        input={"query": query},
        metadata={
            "k": k, "k_per_retriever": k_per_retriever, "rrf_k": rrf_k,
            "n_vector": len(vector_results), "n_bm25": len(bm25_results),
            "n_fused": len(rrf_scores),
            "n_overlap": len(vector_results) + len(bm25_results) - len(rrf_scores),
        },
        output={"top_clauses": [r.get("clause") for r in results]},
    )
    return results
```

Two things in there are easy to get wrong and hard to notice.

**The fusion key must identify a chunk, not approximate one.** The obvious key is
`r["text"][:100]`, and it works right up until two chunks share an opening. This
corpus has exactly that - a run of SOPs that all begin "INTRODUCTION\nSpices,
Inc. has established, documented, and implemented procedure..." - so the prefix
key fused two distinct chunks into one entry and silently dropped a result from
the output. Nothing errors; you just get `k-1` chunks and never know. The
retrievers read different stores and share no id (Chroma's vector results carry
no chunk id back through `vanilla.retrieve`), so the join key is source, page and
the *whole* text.

**`k_per_retriever` has to scale with `k`.** A fixed 10 while `ask.py` asks for
`TOP_K=14` means the fusion is choosing 14 results out of 10 vector candidates -
fewer than the vanilla retriever sees at the same `k`. Any comparison against
vanilla is then confounded: hybrid loses on depth it was never given. Depth is
nearly free here, since both retrievers run off the single embedding call and
BM25 scores the whole corpus regardless, so `max(2k, 20)` is the right default.

### Test the tokenizer first

```bash
uv run python -c "
from retrievers.hybrid import _tokenize
print(_tokenize('clause 2.5.5 internal audits'))
"
# ['clause', '2.5.5', 'internal', 'audits']
```

If `2.5.5` comes back as three tokens, the regex did not take and BM25 will be
useless on exactly the queries it should be best at.

### Compare strategies by hand before evaluating

```bash
RETRIEVAL_STRATEGY=vanilla uv run python ask.py "What does clause 2.5.5 require?"
RETRIEVAL_STRATEGY=hybrid  uv run python ask.py "What does clause 2.5.5 require?"
```

The clause-number query is where you should see the clearest difference.

```bash
git add retrievers/hybrid.py
git commit -m "Week 4 Day 10: hybrid BM25 + cosine with clause-safe tokenizer"
```

---

## Day 11 (Thu): Reranking

### Concept: What reranking adds

After hybrid gives you 10-20 candidates, a reranker runs a more expensive model
over (query, candidate) pairs to reorder them. It does not search; it judges. It
says "of these 20, here is the order I would put them in for this specific
query."

Why not use it for everything? Cost. A cross-encoder scores every candidate
individually; you cannot run it over 10,000 chunks. The pattern is cheap
retrieval to 20 candidates, expensive reranking to sort them.

Where it earns its keep on this corpus: SQF documents repeat "records,"
"verification," "approved," "monitoring," and "documented" across dozens of
clauses. Both retrievers return near-duplicates that differ only in which
requirement they attach to. The cross-encoder reads the full query against each
candidate and disambiguates.

The free-tier twist that other strategies do not hit: rerank's cost is the whole
candidate set, not one query. Voyage caps an unbilled account at 3 RPM *and 10K
TPM*, and a rerank request pays for every candidate it scores. Sending too many
candidates raises a `RateLimitError` that no backoff can clear - a request larger
than the per-minute budget can never succeed. So candidates are trimmed to a
token budget rather than a fixed count, because chunks in this corpus run a 7x
size spread and any fixed count is either wasteful on small chunks or over the
ceiling on large ones. And the rerank call goes through `paced_call` from
`embed.py`, not a fresh client, so it queues behind the same gate the query
embedding does (rerank spends two Voyage calls per question).

### Project: `retrievers/rerank.py`

Complete file. Model name as of writing is `rerank-2.5`; check
https://docs.voyageai.com/docs/reranker if it has been renamed.

```python
"""Hybrid retrieval + Voyage rerank.

Reranking is the one strategy whose cost is a whole candidate set rather than a
single query, so the free tier's token ceiling binds here in a way it does not
elsewhere. Voyage caps an unbilled account at 3 RPM *and 10K TPM*, and a rerank
request pays for every candidate it scores. Sending 56 candidates raised a
RateLimitError that no amount of backoff could clear - a request larger than
the per-minute budget can never succeed, so it retried for 250s and then failed.

Candidates are therefore trimmed to a token budget rather than a fixed count.
Chunks in this corpus run from ~349 characters at the median to 2400 at p95, a
7x spread, so any fixed count is either wasteful on small chunks or over the
ceiling on large ones: 20 large chunks is ~12K tokens, already past the limit.
"""
import os

from langfuse import observe

# Not a fresh voyageai.Client(): rerank spends a request from the same account
# rate limit the query embedding does, so it goes through the shared gate in
# embed.py. See paced_call's docstring for why that matters at eval scale.
from retrievers.embed import paced_call
from retrievers.hybrid import hybrid_retrieve
from tracing import langfuse

RERANK_MODEL = "rerank-2.5"

# Sized for sustained eval throughput, not for one question. Pacing applies per
# Voyage call and rerank spends two per question (embedding + rerank), so the
# floor is 2 x 21s = 42s, or ~1.43 questions/minute against a 10K TPM ceiling.
#
# Measured over the 39 golden questions, taking BM25's top 56 as the candidate
# set:
#
#   budget  kept (median/min)  questions under k=14  sustained TPM
#   6000        18 / 13               3 / 39             8,275
#   7000        21 / 15               0 / 39             9,679
#   8000        24 / 18               0 / 39            11,102   over ceiling
#
# 7000 never truncates below k and still fits, but only by 3%, and that margin
# rests on the estimate below. This corpus is dense with clause numbers like
# 2.5.5.1, which tokenize far worse than prose, so the real count runs above a
# 4-chars-per-token guess and 7000 would likely breach in practice. 6000 costs
# one excerpt on 3 of 39 questions (13 rather than 14) and keeps real headroom.
#
# This is the knob that buys rerank depth, not k_pre_rerank on its own. Raise it
# once the account has a payment method and the 3 RPM / 10K TPM cap is gone.
RERANK_TOKEN_BUDGET = int(os.getenv("VOYAGE_RERANK_TOKEN_BUDGET", "6000"))

# Rough English ratio. Deliberately an estimate: the point is to stay under a
# ceiling, and paying an extra API call to count tokens exactly would spend the
# very budget being measured.
CHARS_PER_TOKEN = 4


def _fit_token_budget(candidates: list[dict], budget: int) -> tuple[list[dict], int]:
    """Take candidates in rank order until the token budget is spent.

    Always keeps at least one, so a single oversized chunk degrades to a
    one-document rerank instead of an empty request Voyage would reject.
    """
    kept: list[dict] = []
    used = 0
    for c in candidates:
        cost = len(c["text"]) // CHARS_PER_TOKEN + 1
        if kept and used + cost > budget:
            break
        kept.append(c)
        used += cost
    return kept, used


@observe(name="rerank-model", as_type="retriever")
def _rerank_call(query: str, documents: list[str], top_k: int):
    """The Voyage call on its own span.

    Separated from the enclosing "rerank" span so the reranker's latency is
    readable against the retrieval that fed it - otherwise a slow question looks
    equally attributable to Chroma, BM25 or Voyage.
    """
    response = paced_call("rerank", query=query, documents=documents,
                          model=RERANK_MODEL, top_k=top_k)
    langfuse.update_current_span(
        input={"query": query, "n_documents": len(documents)},
        metadata={"model": RERANK_MODEL, "top_k": top_k},
        output={"scores": [round(r.relevance_score, 4) for r in response.results]},
    )
    return response


@observe(name="rerank", as_type="retriever")
def rerank_retrieve(query: str, k: int = 5,
                    k_pre_rerank: int | None = None) -> list[dict]:
    """Get candidates from hybrid, rerank with Voyage, return the top k.

    k_pre_rerank is the ceiling on how many candidates are *considered*; on the
    free tier RERANK_TOKEN_BUDGET is what actually decides the depth, since a
    request over 10K TPM fails no matter how few candidates it names. It
    defaults to max(4k, 40) so that a billed account with the budget raised gets
    real headroom - reranking 14 out of 20 has only six candidates to discard,
    which is rarely enough to change the answer.
    """
    if k_pre_rerank is None:
        k_pre_rerank = max(4 * k, 40)

    candidates = hybrid_retrieve(query, k=k_pre_rerank)
    if not candidates:
        # Voyage rejects an empty document list, and there is nothing to rank.
        return []

    n_retrieved = len(candidates)
    candidates, est_tokens = _fit_token_budget(candidates, RERANK_TOKEN_BUDGET)
    response = _rerank_call(query, [c["text"] for c in candidates], top_k=k)

    # Voyage returns indices into the input list along with relevance scores.
    # Carry the clause metadata across or citation breaks at the last hop.
    results = [
        {
            "text": candidates[r.index]["text"],
            "source": candidates[r.index]["source"],
            "clause": candidates[r.index].get("clause"),
            "clause_title": candidates[r.index].get("clause_title"),
            "page": candidates[r.index].get("page"),
            "doc_type": candidates[r.index].get("doc_type"),
            "rank": rank + 1,
            "score": r.relevance_score,
            "original_rank": candidates[r.index].get("rank"),
        }
        for rank, r in enumerate(response.results)
    ]
    langfuse.update_current_span(
        input={"query": query, "n_candidates": len(candidates)},
        metadata={
            "k": k, "k_pre_rerank": k_pre_rerank, "model": RERANK_MODEL,
            "n_retrieved": n_retrieved, "n_reranked": len(candidates),
            "est_tokens": est_tokens, "token_budget": RERANK_TOKEN_BUDGET,
            # True when the budget, not k_pre_rerank, set the depth. Worth
            # filtering on: it also means fewer than k excerpts reached the
            # model if the budget cut below k.
            "budget_trimmed": len(candidates) < n_retrieved,
            "returned_fewer_than_k": len(results) < k,
        },
        output={
            "top_clauses": [r["clause"] for r in results],
            "rank_changes": [(r["original_rank"], r["rank"]) for r in results],
        },
    )
    return results
```

### Test

```bash
RETRIEVAL_STRATEGY=rerank uv run python ask.py "What records must be kept for training?"
```

In Langfuse, open the `rerank` span and read `rank_changes`. Pairs like
`(14, 1)` are the reranker earning its cost: a chunk that hybrid ranked 14th was
actually the best answer. Check `budget_trimmed` in the span metadata too - when
true, the token budget (not `k_pre_rerank`) set the depth, and if it cut below
`k` the answer saw fewer than `k` excerpts.

```bash
git add retrievers/rerank.py
git commit -m "Week 4 Day 11: Voyage reranking over hybrid candidates"
```

---

## Day 12 (Fri): Eval All Three Strategies

The payoff.

```bash
RETRIEVAL_STRATEGY=vanilla uv run python evals/run_eval.py strategy_vanilla "cosine top_k=5"
RETRIEVAL_STRATEGY=hybrid  uv run python evals/run_eval.py strategy_hybrid  "BM25 + cosine + RRF, k=5"
RETRIEVAL_STRATEGY=rerank  uv run python evals/run_eval.py strategy_rerank  "hybrid candidates + Voyage rerank-2.5"
```

About 6-10 minutes and $1 each, $3 total.

```bash
uv run python evals/analyze.py compare_three \
  evals/results/TIMESTAMP_strategy_vanilla.csv \
  evals/results/TIMESTAMP_strategy_hybrid.csv \
  evals/results/TIMESTAMP_strategy_rerank.csv
```

### How to read the table

`compare_three` prints compliance metrics before judged axes. Read in that order.

**Probe refusal rate.** This must not fall as retrieval improves. The failure
mode to watch for: better retrieval surfaces a loosely-related clause for an
out-of-corpus question, and the model treats "something relevant came back" as
permission to answer. If refusal drops from 5/5 to 3/5 while `overall` rises
0.2, that is a regression, not an improvement, and it is the single most
important thing this eval can tell you.

**clause hit@3.** Your cleanest retrieval signal. Expect hybrid to beat vanilla
by a wide margin on this corpus, concentrated in the areas where exact
terminology matters. If hybrid does not beat vanilla here, check the tokenizer
test from Day 10 before believing the result.

**Citation correctness and grounding.** Did better retrieval actually let the
model cite the right clause and stop it reaching for outside knowledge?

**Then the standard three axes**, which for this corpus are the least
load-bearing numbers in the table.

### Save the summary

```bash
mkdir -p evals/results
# Hand-copy the summary tables into a tracked file rather than committing raw CSVs
$EDITOR evals/results/three_way_summary.md
git add -f evals/results/three_way_summary.md
git commit -m "Week 4 Day 12: three-way retrieval strategy comparison"
```

---

## Day 13 (Sat): Error Analysis + Blog Draft

### Error analysis pass

Pull every scored question where rerank still misses (`hit_3` false or `citation`
below 3) and bucket the cause. On this corpus they cluster into four:

1. **Requirement split across two clauses.** The answer needs 2.5.3 and 2.5.5;
   chunking put them far apart and only one was retrieved. Traces to Week 2 Day 10.
2. **Clause number in the query not matched.** Traces to the tokenizer, Week 4 Day 10.
3. **Requirement trapped in a table** that extraction flattened badly. Traces to
   Week 2 Day 9; this is where selectively using pdfplumber on that one file pays.
4. **Genuine corpus gap.** The requirement is not in your indexed documents.
   Refusal is correct; this is not an error and should not be counted as one.

Naming the taxonomy is itself a portfolio artifact. It shows you can debug a
retrieval system rather than just assemble one. Put it in `EVAL_REPORT.md` and in
`NOTES.md`.

### Update `EVAL_REPORT.md`

Add the three-way comparison table, the error taxonomy, and a short section on
what you would do next given the failure distribution.

### Draft the blog post

Working title: "Retrieval Strategies for a Compliance RAG: Citing the Right
Clause." The material that makes this post worth reading, none of which appears
in a generic docs-bot writeup:

- Ground truth as a clause reference, which gives a judge-free retrieval metric
- Why default tokenization destroys clause numbers, with the before/after
- The refusal probe set, and refusal rate as a first-class metric
- The three-way comparison led by citation accuracy rather than answer quality
- The error taxonomy

---

## Day 14 (Sun): Publish + Wrap

### Update the README

```markdown
## Architecture

data/source/*.pdf,docx
  -> ingest.py     extract, OCR scanned pages, strip running headers
  -> data/raw/*.jsonl
  -> chunk.py      split on SQF clause boundaries, carry clause/page metadata
  -> data/chunks.jsonl
  -> embed.py      Voyage voyage-3-lite -> Chroma (sqf_docs)
  -> ask.py        retrieve -> assemble cited prompt -> Claude Sonnet 4.6

Retrieval strategies (RETRIEVAL_STRATEGY env var):
  vanilla  cosine similarity, top_k=14
  hybrid   BM25 + cosine fused with RRF (clause-safe tokenizer)
  rerank   hybrid candidates reranked by Voyage rerank-2.5

Evaluation:
  evals/golden.jsonl     34 scored questions + 5 out-of-corpus refusal probes
  evals/validate.py      golden-set schema, clause grounding, area coverage
  evals/metrics.py       deterministic clause_hit@k, refusal, citation grounding
  evals/check_metrics.py regression cases for metrics.py
  evals/judge.py         Claude Sonnet 4.6, five axes including citation and grounding
  evals/run_eval.py      runner, CSV output, Langfuse scores
  evals/analyze.py       summarize / compare / compare_three

Observability: Langfuse, self-hosted. Every run is a session; every question is
a trace with judge scores attached.
```

Note in the README that `data/source/` is gitignored: the repo ships the pipeline,
not the certification documents.

### The Pipeline Drawing Test

Before calling Week 4 done, draw from memory: the ingestion path, the three
retrieval strategies and where they diverge, where the clause metadata enters and
every hop it has to survive, and how a judge score gets from `judge.py` to the
Langfuse UI. If any leg is fuzzy, reread that file.

### Week 4 Wrap-up Checklist

- [ ] Langfuse running, traces appearing with correct parent/child nesting
- [ ] `_tokenize("clause 2.5.5")` preserves `2.5.5` as one token
- [ ] Clause metadata survives all three retrievers into the prompt
- [ ] Three-way comparison run and saved
- [ ] Probe refusal rate stable or better under rerank
- [ ] Error taxonomy written up
- [ ] `EVAL_REPORT.md` and README updated, repo pushed

---

## A Note on Pacing

Week 3 Days 1-2 are the ones people rush and should not. The golden set is the
measuring instrument for everything after it, and a set with invented clause
numbers produces confident numbers that mean nothing. If you fall behind, cut the
Day 6 variant experiment before you cut golden-set quality.

The clause numbers and reference answers in this document are placeholders.
Replace every one of them from your actual documents before you trust a single
number the harness prints.
