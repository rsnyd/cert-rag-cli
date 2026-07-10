# Week 2 - Evaluation harness and observability

## Objective

Build the golden question set, the LLM-as-judge harness, and Langfuse tracing
over the SQF baseline. Same scaffolding as the Drupal build. What changes is the
ground truth (clauses, not free text), the judge rubric (citation correctness
and no-fabrication), and a trace tag so you do not mix corpora.

## What ports as-is

The judge runner, the metrics aggregation, the Langfuse client setup, and the
trace/span decorators all port directly. Do not rewrite them. The sections below
are the only edits.

## Step 1 - golden set with clause ground truth

In the Drupal build your ground truth was likely an expected free-text answer.
For SQF, make the expected answer a clause reference. This gives you a cleaner,
cheaper retrieval metric (did the correct clause appear in top-k?) that does not
depend on the judge, alongside the judged answer-quality score.

`eval/golden.yaml`

```yaml
- id: ia-frequency
  question: "How often must internal audits be conducted under the SQF System?"
  expected_clause: "2.4.3"          # set to the real clause in YOUR edition
  category: system_elements

- id: recall-mock
  question: "How frequently must the recall and withdrawal program be tested?"
  expected_clause: "2.5.4"
  category: traceability_recall

- id: practitioner-role
  question: "What is the SQF practitioner responsible for?"
  expected_clause: "2.1.2"
  category: management_commitment

- id: supplier-approval
  question: "What are the requirements for approving raw material suppliers?"
  expected_clause: "2.3.3"
  category: supplier_approval

- id: food-defense
  question: "What must the food defense plan contain?"
  expected_clause: "2.7.1"
  category: food_defense

- id: training-records
  question: "What training records must be maintained for personnel?"
  expected_clause: "2.9.2"
  category: training

- id: ccp-monitoring
  question: "How must critical control points be monitored?"
  expected_clause: "2.4.5"
  category: food_safety_plan

- id: corrective-action
  question: "What is required when a critical limit is exceeded?"
  expected_clause: "2.5.3"
  category: corrective_action

- id: document-control
  question: "What are the requirements for controlling SQF System documents?"
  expected_clause: "2.2.1"
  category: document_control

- id: complaint-handling
  question: "How must customer complaints be managed and recorded?"
  expected_clause: "2.6.1"
  category: complaints

- id: verification-schedule
  question: "What verification activities must be scheduled and by whom?"
  expected_clause: "2.5.2"
  category: verification

- id: mgmt-review
  question: "How often must management review the SQF System?"
  expected_clause: "2.1.4"
  category: management_review
```

The clause values above are placeholders keyed to typical Module 2 structure.
Replace each `expected_clause` with the real clause from your documents before
you trust the retrieval metric. Expand to 30 the way you did for Drupal, keeping
the category spread so you can see which requirement areas retrieve poorly.

## Step 2 - retrieval metric (new, cheap, judge-free)

Because ground truth is a clause, you can score retrieval directly.

```python
def clause_hit_at_k(retrieved: list[dict], expected_clause: str, k: int) -> bool:
    return any(
        (r.get("clause") or "").startswith(expected_clause)
        for r in retrieved[:k]
    )
```

`startswith` so that citing `2.4.3.1` counts as a hit for expected `2.4.3`.
Report hit@1, hit@3, hit@5 per category. This becomes your primary signal in
Weeks 3 and 4 when you compare retrieval strategies, and it does not cost a judge
call.

## Step 3 - judge rubric

Port the judge runner unchanged; swap the rubric. Add two dimensions that a
compliance answer lives or dies on.

```text
Score the ANSWER against the QUESTION and the retrieved CONTEXT on four
dimensions, 1-5 each. Return JSON only.

1. correctness: Does the answer state the requirement accurately per CONTEXT?
2. citation_correctness: Does it cite the clause that actually contains the
   requirement? A confident answer citing the wrong clause scores 1-2 here even
   if the prose is right.
3. grounding: Is every claim supported by CONTEXT with no invented requirement,
   number, or frequency? Any fabricated specific (a made-up audit frequency, a
   made-up temperature) caps this dimension at 2.
4. appropriate_refusal: If CONTEXT lacked the requirement, did the answer refuse
   rather than guess? If CONTEXT contained it, score 5 by default.

Return: {"correctness": n, "citation_correctness": n, "grounding": n,
"appropriate_refusal": n, "rationale": "..."}
```

Weight `citation_correctness` and `grounding` heavily in your composite. In this
domain a fluent answer with a wrong clause is a worse outcome than a refusal, and
your scoring should reflect that ordering.

## Step 4 - a small fabrication probe set

Add a handful of questions whose answers are deliberately NOT in the corpus (for
example a requirement from a different GFSI scheme, or an invented threshold).
The correct behavior is refusal. Track refusal rate on this set separately. It is
the single most important safety metric for a compliance RAG and it makes a
strong talking point.

## Step 5 - Langfuse (port, with one tag)

Wiring is unchanged. Two edits:

- Tag every trace `corpus=sqf` (and optionally `edition=<n>`) so these traces do
  not blur into the frozen Drupal traces in the same Langfuse project. Or point
  at a separate Langfuse project entirely.
- Attach `expected_clause` and the computed `clause_hit_at_k` as trace metadata,
  so a failing eval row is one click from the retrieved chunks that caused it.

```python
langfuse.trace(
    name="ask",
    tags=["corpus:sqf", f"edition:{edition}"],
    metadata={"expected_clause": expected_clause, "hit_at_5": hit},
)
```

## Definition of done

- `uv run cert-rag eval` produces per-category correctness, citation-correctness,
  grounding, and refusal scores plus hit@{1,3,5}.
- The fabrication probe set reports a refusal rate, and it is high.
- Langfuse shows this run tagged `corpus:sqf`, and clicking a low-scoring row
  surfaces the retrieved clauses.

## Portfolio note

The composite you want to show later is not "answer quality went up." It is
"citation accuracy and refusal rate held while retrieval improved." Start
recording those two numbers now so Week 4's comparison has a baseline.
