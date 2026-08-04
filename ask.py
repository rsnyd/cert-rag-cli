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