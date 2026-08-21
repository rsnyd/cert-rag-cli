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
obvious. Mix difficulties: 10 easy (single clause, single lookup), 10 medium
(synthesis across 2-3 clauses), 10 hard (edge cases, comparisons, "what happens
if" reasoning).

On top of the 30, add 5 probe questions whose answers are deliberately not in
your corpus. These measure refusal, and they are the single most important safety
metric you will produce.

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
| `id` | `Q001`-`Q030` for scored questions, `P001`+ for probes |
| `difficulty` | `easy`, `medium`, `hard`, or `probe` |
| `tags` | 1-3 topic tags, used to slice results later |
| `question` | What the user asks |
| `expected_clause` | The clause that contains the requirement. `null` for probes |
| `expect_refusal` | `true` only for probes |
| `reference_answer` | What a competent SQF practitioner would accept |
| `expected_topics` | 3-6 concrete terms that should appear |

> **These examples are scaffolding, not content.** The clause numbers and
> reference answers below are plausible-looking placeholders. SQF clause
> numbering differs by edition and by which code you are certified against, and
> your internal SOPs have their own numbering entirely. Open your actual
> documents and rewrite every `expected_clause` and every `reference_answer`
> from what they say. If you skip this, the retrieval metric measures nothing
> and the judge scores against fiction.

```json
{"id": "Q001", "difficulty": "easy", "tags": ["internal-audit", "verification"], "question": "How often must internal audits of the SQF System be conducted?", "expected_clause": "2.5.5", "expect_refusal": false, "reference_answer": "Internal audits of the SQF System must be conducted at least annually. The audit schedule and responsibility are defined by the SQF practitioner, and audit findings, corrective actions, and follow-up verification are documented and the records retained.", "expected_topics": ["annually", "internal audit", "corrective action", "records", "schedule"]}
{"id": "Q002", "difficulty": "easy", "tags": ["management-commitment"], "question": "What is the SQF practitioner responsible for?", "expected_clause": "2.1.2", "expect_refusal": false, "reference_answer": "The SQF practitioner is designated by senior management, must be employed full time by the site, and is responsible for developing, implementing, reviewing and maintaining the SQF System. The practitioner must have completed a HACCP training course and be able to demonstrate competency in food safety principles relevant to the site.", "expected_topics": ["designated", "full time", "HACCP training", "implement and maintain", "competency"]}
{"id": "Q003", "difficulty": "medium", "tags": ["recall", "traceability"], "question": "How frequently must the product withdrawal and recall program be tested, and what has to be recorded?", "expected_clause": "2.6.3", "expect_refusal": false, "reference_answer": "The withdrawal and recall program must be tested at least annually to verify it is effective. The test is documented, including the reconciliation of product quantities, the time taken to complete the trace, and any gaps found. Results are reviewed and used to update the program.", "expected_topics": ["annually", "mock recall", "effectiveness", "reconciliation", "records"]}
{"id": "Q004", "difficulty": "medium", "tags": ["supplier-approval", "specifications"], "question": "What is required to approve a raw material supplier?", "expected_clause": "2.3.3", "expect_refusal": false, "reference_answer": "Raw materials and packaging must be sourced from approved suppliers. Approval is based on a documented risk assessment and supported by agreed specifications and evidence of the supplier's food safety controls, such as a certificate of conformance or an audit. An approved supplier register is maintained and supplier performance is reviewed at a defined frequency.", "expected_topics": ["approved supplier register", "risk assessment", "specifications", "certificate of conformance", "review"]}
{"id": "Q005", "difficulty": "hard", "tags": ["corrective-action", "food-safety-plan"], "question": "What has to happen when a critical limit at a CCP is exceeded?", "expected_clause": "2.5.3", "expect_refusal": false, "reference_answer": "When a critical limit is exceeded the affected product must be identified, segregated and held pending disposition. The cause is investigated, corrective action is taken to bring the CCP back under control, and preventative action is taken to stop recurrence. The deviation, the product disposition, the responsible person and the verification of effectiveness are all recorded.", "expected_topics": ["hold and segregate", "root cause", "corrective action", "preventative action", "disposition", "records"]}
```

Three probe questions to add now, so refusal is measured from the first run:

```json
{"id": "P001", "difficulty": "probe", "tags": ["probe", "out-of-corpus"], "question": "What is the maximum civil penalty for an OSHA recordkeeping violation?", "expected_clause": null, "expect_refusal": true, "reference_answer": "Not found in the provided documents. This is an occupational safety question and is outside the SQF certification corpus.", "expected_topics": []}
{"id": "P002", "difficulty": "probe", "tags": ["probe", "other-scheme"], "question": "What does BRCGS require for the pre-production hygiene check?", "expected_clause": null, "expect_refusal": true, "reference_answer": "Not found in the provided documents. BRCGS is a different GFSI scheme and its requirements are not in this corpus.", "expected_topics": []}
{"id": "P003", "difficulty": "probe", "tags": ["probe", "invented-threshold"], "question": "What is the required minimum ozone concentration for the water reuse loop?", "expected_clause": null, "expect_refusal": true, "reference_answer": "Not found in the provided documents. No such threshold is specified in the corpus.", "expected_topics": []}
```

Note what makes a good probe: it sounds like it belongs. "What is the maximum
fine for jaywalking" is not a useful probe because no system would answer it. A
question about a neighbouring GFSI scheme, or a plausible-sounding threshold that
simply is not in your documents, is exactly the kind of thing a RAG will
confabulate when retrieval returns something adjacent.

```bash
git add evals/
git commit -m "Week 3 Day 1: eval scaffolding and first golden questions"
```

---

## Day 2 (Tue): Finish the Golden Set + Metrics

The slog day. The output is your most valuable artifact for the rest of the plan.

### Process

Block 2-3 uninterrupted hours. Write the remaining 25 scored questions and 2 more
probes in one sitting if you can. Work from your actual documents, open beside
you. Aim for 10 easy, 10 medium, 10 hard, plus 5 probes.

Tag every question. Useful tags for this corpus:

`management-commitment`, `management-review`, `document-control`, `records`,
`specifications`, `supplier-approval`, `food-safety-plan`, `ccp`,
`corrective-action`, `verification`, `internal-audit`, `traceability`, `recall`,
`food-defense`, `food-fraud`, `allergen-management`, `training`, `calibration`,
`sanitation`, `pest-control`

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
  requirement. If a requirement genuinely spans two clauses, use the primary one;
  the `startswith` matching in the metric handles sub-clauses automatically.

### Coverage checklist

Hit at least one question in each area:

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

Complete file. Create it and paste:

```python
"""Validate the structure of the SQF golden set.

Checks the scored questions and the refusal probes separately, since they have
different required fields: a scored question needs an expected_clause, a probe
needs expect_refusal set and no clause.
"""
import json
from collections import Counter
from pathlib import Path

GOLDEN_FILE = Path("evals/golden.jsonl")

REQUIRED = {
    "id", "difficulty", "tags", "question",
    "expected_clause", "expect_refusal",
    "reference_answer", "expected_topics",
}

SCORED_DIFFICULTIES = {"easy", "medium", "hard"}


def main() -> None:
    records = [json.loads(line) for line in GOLDEN_FILE.open(encoding="utf-8")]

    # Every record carries every field, so the runner never has to use .get()
    for r in records:
        missing = REQUIRED - r.keys()
        assert not missing, f"{r.get('id')} missing fields: {missing}"

    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids), "Duplicate IDs"

    scored = [r for r in records if r["difficulty"] in SCORED_DIFFICULTIES]
    probes = [r for r in records if r["difficulty"] == "probe"]

    unknown = {r["difficulty"] for r in records} - SCORED_DIFFICULTIES - {"probe"}
    assert not unknown, f"Invalid difficulty values: {unknown}"

    assert len(scored) == 30, f"Expected 30 scored questions, got {len(scored)}"
    assert len(probes) >= 3, f"Expected at least 3 probes, got {len(probes)}"

    # Scored questions must name a clause; probes must not.
    for r in scored:
        assert r["expected_clause"], f"{r['id']} has no expected_clause"
        assert r["expect_refusal"] is False, f"{r['id']} is scored but expects refusal"
    for r in probes:
        assert r["expected_clause"] is None, f"{r['id']} is a probe but names a clause"
        assert r["expect_refusal"] is True, f"{r['id']} is a probe but expect_refusal is False"

    difficulties = Counter(r["difficulty"] for r in scored)
    print(f"Scored distribution: {dict(difficulties)}")
    for level in SCORED_DIFFICULTIES:
        assert difficulties[level] == 10, (
            f"Expected 10 {level} questions, got {difficulties[level]}"
        )

    tags = Counter()
    for r in records:
        tags.update(r["tags"])
    print(f"Top tags: {tags.most_common(8)}")

    clauses = Counter(r["expected_clause"] for r in scored)
    dupes = {c: n for c, n in clauses.items() if n > 2}
    if dupes:
        print(f"Note: clauses used more than twice: {dupes}")

    print(f"\n{len(scored)} scored questions and {len(probes)} probes validated.")


if __name__ == "__main__":
    main()
```

```bash
uv run python evals/validate.py
```

Expected output:

```
Scored distribution: {'easy': 10, 'medium': 10, 'hard': 10}
Top tags: [('verification', 6), ('records', 5), ('food-safety-plan', 4), ...]

30 scored questions and 5 probes validated.
```

### Project: `evals/metrics.py`

The deterministic metrics. No LLM call, no cost, no drift. Complete file:

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
```

```bash
uv run python evals/validate.py
git add evals/golden.jsonl evals/validate.py evals/metrics.py
git commit -m "Week 3 Day 2: 30-question SQF golden set, 5 refusal probes, deterministic metrics"
```

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

If they are present but wrong, the credentials are checked once here rather than
discovered span by span, and TRACING_ENABLED says which of the three states this
process is in: keys absent, keys broken, or tracing live.

The decorated functions live in ask.py, evals/judge.py and evals/run_eval.py.
This module only owns client construction, so there is a single place that tags
every trace with the git commit (release).
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
    exports in a background batch: a bad key surfaces as a per-span 401 on stderr
    that no caller ever sees, so a long run happily finishes having sent nothing.
    """
    try:
        response = requests.get(
            f"{host.rstrip('/')}/api/public/projects",
            auth=(public_key, secret_key), timeout=5,
        )
    except requests.RequestException:
        return None
    return response.status_code == 200


_TRACING_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
AUTH_OK: bool | None = None

if _TRACING_ON:
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
        print(
            "\n" + "=" * 72 + "\n"
            f"LANGFUSE TRACING DISABLED - {_reason}.\n"
            f"  host: {_host}\n"
            "Everything else still runs; only tracing is off. Fix the LANGFUSE_*\n"
            "values in .env (see .env.example) and re-run.\n"
            + "=" * 72 + "\n",
            file=sys.stderr,
        )
        # Switch the SDK off outright instead of letting it retry every span.
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logging.getLogger("langfuse").setLevel(logging.CRITICAL)
else:
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

FACTUAL CORRECTNESS - are the claims the answer actually makes correct?
Judge only the claims present; missing points are handled under Completeness, and
a terse-but-correct answer still scores 5 here.
  5 = Every claim is accurate. No false or misleading statements.
  4 = Accurate apart from one minor imprecision that would not mislead a practitioner.
  3 = Mostly correct but contains one significant inaccuracy someone could act on wrongly.
  2 = Multiple significant inaccuracies, or a central claim is wrong.
  1 = The core requirement is wrong, or a frequency, threshold or responsibility is
      misstated in a way that would fail an audit (a program described as quarterly
      when the requirement is annual).

COMPLETENESS - how many of the reference answer's key points are covered?
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

GROUNDING - is every claim supported by the excerpts, with nothing invented?
  5 = Every claim traces to the excerpts. Nothing added from outside knowledge.
  4 = Fully supported apart from one harmless general statement.
  3 = Mostly supported, but one claim goes beyond what the excerpts say.
  2 = Contains a fabricated specific: a frequency, threshold, temperature,
      responsibility or record requirement that appears nowhere in the excerpts.
  1 = Substantially invented, or answers from general food-safety knowledge when
      the excerpts do not contain the requirement.
Any fabricated specific caps this axis at 2 regardless of how much else is right.

Calibration:
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
and adds the strategy switch that Weeks 4 Days 10-11 will use.

Replace `ask.py` entirely with:

```python
"""Ask a question. Retrieve context. Generate a cited answer."""
import os
import sys

from anthropic import Anthropic
from langfuse import observe, propagate_attributes

# Importing tracing constructs the Langfuse client from the LANGFUSE_* env vars.
# If those are unset the SDK disables itself and every @observe below is a no-op.
from tracing import langfuse

# Retrieved chunks per question. 14 rather than 5 because clause chunks are
# small (~690 chars): at k=5 the model saw ~3.4K chars and completeness was
# the weakest axis. A 5/10/14 sweep moved it 3.85 -> 4.09 -> 4.24 (paired
# sign test p=0.013) with clause citation flat at 97%. Not higher: relevance
# falls monotonically over the same sweep as tangential excerpts dilute the
# answer. 14 is the knee, not a ceiling. Env-overridable so the eval can sweep it.
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


@observe(name="rag-answer")
def answer_question_with_context(query: str) -> tuple[str, list[dict]]:
    """Run the full pipeline and return the answer AND the chunks it used.

    The eval harness needs the chunks: citation correctness and grounding can
    only be judged against the context the model actually saw, and the clause
    hit metric is computed from it directly.
    """
    chunks = retrieve(query)
    prompt = assemble_prompt(query, chunks)
    return generate(prompt), chunks


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
from evals.metrics import clause_hit_at_k, is_refusal, retrieved_clauses
from langfuse import propagate_attributes
from tracing import langfuse

# The five judge axes, mapped to the score names they get in Langfuse.
SCORE_AXES = (
    "factual_correctness", "completeness", "relevance",
    "citation_correctness", "grounding",
)

GOLDEN_FILE = Path("evals/golden.jsonl")
RESULTS_DIR = Path("evals/results")
RESULTS_DIR.mkdir(exist_ok=True)

# Every row carries every column so the CSV is rectangular even though probe
# rows have no judge scores.
FIELDNAMES = [
    "id", "difficulty", "tags", "question", "expected_clause",
    "system_answer", "refused",
    "factual", "complete", "relevant", "citation", "grounding", "overall",
    "hit_1", "hit_3", "hit_5", "top_clauses",
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
        "hit_1": "", "hit_3": "", "hit_5": "",
        "top_clauses": "",
        "reasoning": "",
        "elapsed_sec": "",
    }


def run_eval(run_name: str, config_notes: str = "", limit: int | None = None) -> Path:
    """Run the eval, write a CSV, print a summary.

    If `limit` is set, only the first N golden records are run - handy for cheap
    smoke tests that stay under the Voyage free-tier rate limit.
    """
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

            try:
                scores = judge(rec["question"], rec["reference_answer"],
                               system_answer, chunks)
            except Exception as e:
                langfuse.update_current_span(level="ERROR", status_message=f"JUDGE: {e}")
                print(f"  {i:2d}/{len(records)} {rec['id']} JUDGE ERROR: {e}")
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
    print_summary(rows)

    # Force-send buffered traces and scores before the script exits.
    langfuse.flush()
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        print(f"\nTraces sent to Langfuse under session '{run_session}'.")
    return out_file


def print_summary(rows: list[dict]) -> None:
    """Print the compliance metrics first, then the judged axes."""
    scored = [r for r in rows if r["difficulty"] != "probe" and r["overall"] != ""]
    probes = [r for r in rows if r["difficulty"] == "probe"]

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
uv run python evals/run_eval.py baseline "vanilla cosine, top_k=5, clause chunking"
```

Roughly 6-10 minutes and about $1: 35 RAG calls plus 30 judge calls, sequential.
Output looks like:

```
Running eval: baseline (35 records, strategy=vanilla)
Config: vanilla cosine, top_k=5, clause chunking

   1/35 Q001 [easy  ] F=5 C=4 R=5 Cite=5 G=5 avg=4.80 hit@3=Y
   2/35 Q002 [easy  ] F=4 C=3 R=5 Cite=3 G=4 avg=3.80 hit@3=Y
   3/35 Q003 [medium] F=3 C=2 R=4 Cite=2 G=3 avg=2.80 hit@3=n
   ...
  31/35 P001 [probe ] REFUSED (pass)
  32/35 P002 [probe ] ANSWERED (FAIL)
  ...

=== Refusal probes (n=5) ===
  Correctly refused:  4/5  (80%)

=== Retrieval (n=30) ===
  clause hit@1:       43%
  clause hit@3:       67%
  clause hit@5:       73%
  False refusals:     2/30

=== Judged axes (n=30) ===
  Citation:   3.53
  Grounding:  4.07
  Factual:    3.87
  Complete:   3.43
  Relevant:   4.20
  Overall:    3.82
```

That is your baseline. Read it in the order the summary prints it, which is
deliberate: refusal behavior first, then retrieval, then answer quality.

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

AXES = ["citation", "grounding", "factual", "complete", "relevant", "overall"]
HITS = ["hit_1", "hit_3", "hit_5"]


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
    vals = df["refused"].map(lambda v: str(v).lower() == "true")
    return 100.0 * vals.mean() if len(df) else float("nan")


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
        latest = sorted(Path("evals/results").glob("*.csv"))[-1]
        summarize(str(latest))
```

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
Chroma with clause, page, module and doc_type metadata.

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
| False refusal rate (scored) | <n>/30 |
| clause hit@1 | <n>% |
| clause hit@3 | <n>% |
| clause hit@5 | <n>% |

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

1. Second-reader review of the 30 reference answers and expected clauses
2. Larger golden set (100+) for more stable averages
3. Multiple runs per config to estimate judge variance
4. Hybrid retrieval and reranking (Week 4)
5. Separate the retrieval metric from the generation metric fully, so a failure
   is attributable without reading the CSV
```

Fill in real numbers. Keep it honest. The "what this didn't measure" section is
what distinguishes a real engineer's eval report from theater, and the note about
unreviewed reference answers is the most credible sentence in the document.

### Week 3 Wrap-up Checklist

- [ ] 30 scored questions and 5 probes in `evals/golden.jsonl`, all with real clause numbers from your documents
- [ ] `validate.py` passes
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
# clear failure into a slow one. Matching on the exception TYPE, not on substrings
# of its message, so a server error is not misread as throttling because its text
# happened to contain a number.
RETRYABLE = (
    voyage_error.RateLimitError,
    voyage_error.ServerError,
    voyage_error.ServiceUnavailableError,
    voyage_error.APIConnectionError,
)

# One client for the process. Constructing one per call re-read the environment
# and discarded any connection reuse for no benefit.
_client = None
# None rather than 0.0 means "no call yet". monotonic()'s zero point is undefined,
# so a real reading can legitimately be 0.0 and a truthiness test would silently
# skip the first interval.
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
    spends two requests per question - one embedding, one rerank. Pacing only the
    embedding would let the rerank half slip past the limit unmetered.

    `operation` names the method on the client: "embed", "rerank".

    Retries back off exponentially (10s, 20s, 40s, then capped at 60s). Pacing
    still applies between attempts, so the two compose to max(interval, backoff).

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

CHUNKS_FILE = Path("data/chunks.jsonl")

# Built once at first use and cached for the process.
_chunks_cache: list[dict] | None = None
_bm25_cache: BM25Okapi | None = None

# Keep dotted alphanumerics intact so clause numbers survive tokenization:
#   "clause 2.5.5 internal audits" -> ["clause", "2.5.5", "internal", "audits"]
# A plain \w+ pattern would yield ["clause", "2", "5", "5", ...], destroying the
# most precise lexical signal this corpus has.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)+|[a-z0-9]+")


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
        _chunks_cache = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
        _bm25_cache = BM25Okapi([_tokenize(_index_text(c)) for c in _chunks_cache])
    return _chunks_cache, _bm25_cache


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
def hybrid_retrieve(query: str, k: int = 5, k_per_retriever: int = 10,
                    rrf_k: int = 60) -> list[dict]:
    """Run vector + BM25, fuse via Reciprocal Rank Fusion."""
    vector_results = vector_retrieve(query, k=k_per_retriever)
    bm25_results = bm25_retrieve(query, k=k_per_retriever)

    rrf_scores: dict[str, float] = {}
    seen: dict[str, dict] = {}

    # Derive rank from list position (1-based). The vector retriever returns
    # results nearest-first but with no "rank" key, so don't rely on one.
    for rank, r in enumerate(vector_results, 1):
        key = r["text"][:100]  # prefix as dedup key
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rank + rrf_k)
        seen[key] = r

    for rank, r in enumerate(bm25_results, 1):
        key = r["text"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rank + rrf_k)
        seen.setdefault(key, r)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    results = [
        {**seen[key], "score": score, "rank": rank + 1}
        for rank, (key, score) in enumerate(ranked)
    ]
    langfuse.update_current_span(
        input={"query": query},
        output={"top_clauses": [r.get("clause") for r in results]},
    )
    return results
```

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
elsewhere. A rerank request pays for every candidate it scores, and a request
over 10K TPM fails no matter how few candidates it names, so candidates are
trimmed to a token budget rather than a fixed count.
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

# Sized for sustained eval throughput, not one question. Measured over the 39
# golden questions against BM25's top ~56 candidates, 6000 tokens costs one
# excerpt on 3 of 39 questions and keeps real headroom under 10K TPM; 7000 fits
# only by ~3% and this clause-dense corpus tokenizes worse than that estimate.
# This is the knob that buys rerank depth. Raise it once the 10K TPM cap is gone.
RERANK_TOKEN_BUDGET = int(os.getenv("VOYAGE_RERANK_TOKEN_BUDGET", "6000"))

# Rough English ratio. Deliberately an estimate: paying an API call to count
# tokens exactly would spend the very budget being measured.
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
    """The Voyage call on its own span, so the reranker's latency is readable
    against the retrieval that fed it."""
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

    k_pre_rerank is the ceiling on candidates considered; on the free tier
    RERANK_TOKEN_BUDGET is what actually decides depth. Defaults to max(4k, 40)
    so a billed account with the budget raised gets real headroom.
    """
    if k_pre_rerank is None:
        k_pre_rerank = max(4 * k, 40)

    candidates = hybrid_retrieve(query, k=k_pre_rerank)
    if not candidates:
        return []   # Voyage rejects an empty document list.

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
  vanilla  cosine similarity, top_k=5
  hybrid   BM25 + cosine fused with RRF (clause-safe tokenizer)
  rerank   hybrid candidates reranked by Voyage rerank-2.5

Evaluation:
  evals/golden.jsonl   30 scored questions + 5 out-of-corpus refusal probes
  evals/metrics.py     deterministic clause_hit@k and refusal detection
  evals/judge.py       Claude Sonnet 4.6, five axes including citation and grounding
  evals/run_eval.py    runner, CSV output, Langfuse scores
  evals/analyze.py     summarize / compare / compare_three

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
