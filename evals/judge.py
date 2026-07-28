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