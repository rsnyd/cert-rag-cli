"""Day 5: Agentic RAG with LangGraph. Retrieve -> grade -> (rewrite & retry | generate).

Carries the retrieved chunks in state so the answer function can return them for
the eval, matching ask.answer_question_with_context.

Everything that the Day 6 comparison holds constant is imported, not restated:
the system prompt, the model, k and the excerpt formatter all come from ask.py
via langchain_rag.py. The only thing this file adds is the loop - which is the
whole point of running it as a third implementation.
"""
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

import env  # noqa: F401  - .env before any os.environ read, same as every entry point
from ask import SYSTEM_PROMPT, TOP_K, _score_answer, assemble_prompt, format_excerpts
from langchain_rag import _chat_model, _docs_to_chunks, get_retriever
from tracing import langfuse

# Each retry costs a full retrieval, and under the Voyage pacing gate a
# retrieval is a real 21 seconds. Two attempts is already up to ~60s of waiting
# on a question the grader keeps rejecting; three would make a 39-record eval
# untenable on the free tier.
MAX_ATTEMPTS = 2


# The state carried through the graph
class RAGState(TypedDict):
    question: str
    original_question: str
    context: str
    chunks: list
    answer: str
    attempts: int
    relevant: bool


@lru_cache(maxsize=None)
def _grader():
    """Yes/no relevance judge. Same model as the answer path - a cheaper grader
    would be a reasonable optimization, but it would also mean the loop's
    decisions came from a different model than the comparison is about."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You judge whether the SQF documentation excerpts are "
                   "sufficient to answer the question. Reply with only 'yes' or 'no'."),
        ("human", "Excerpts:\n{context}\n\nQuestion: {question}\n\n"
                  "Are the excerpts sufficient?"),
    ])
    return prompt | _chat_model() | StrOutputParser()


@lru_cache(maxsize=None)
def _rewriter():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's question to be more specific and "
                   "retrieval-friendly for SQF certification documentation. Use the "
                   "vocabulary of the SQF code (clauses, requirements, records, "
                   "verification). Return only the rewritten question."),
        ("human", "{question}"),
    ])
    return prompt | _chat_model() | StrOutputParser()


@lru_cache(maxsize=None)
def _generator():
    """The compliance prompt, imported rather than paraphrased.

    SYSTEM_PROMPT arrives as a SystemMessage for the reason langchain_rag.py
    documents: a ("system", ...) tuple is parsed as a template, so a brace in
    ask.py's prompt would fail here. The human turn is built by
    ask.assemble_prompt, so the model sees the same bytes as the other two
    implementations - including the refusal instruction, which the eval's probe
    records depend on being present verbatim.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "{user_message}"),
    ])
    return prompt | _chat_model() | StrOutputParser()


def retrieve_node(state: RAGState) -> dict:
    docs = get_retriever(k=TOP_K).invoke(state["question"])
    chunks = _docs_to_chunks(docs)
    return {"chunks": chunks, "context": format_excerpts(chunks)}


def grade_node(state: RAGState) -> dict:
    """Ask the model whether the retrieved context can answer the question."""
    verdict = _grader().invoke(
        {"context": state["context"], "question": state["original_question"]}
    )
    return {"relevant": verdict.strip().lower().startswith("yes")}


def rewrite_node(state: RAGState) -> dict:
    """Rewrite the query to retrieve better context, then loop back.

    Rewrites the *current* question, not the original one. Rewriting the
    original every time makes attempt 2 re-issue attempt 1's query almost
    verbatim, so the second retrieval returns the same chunks the grader has
    already rejected and the retry is 21 seconds of nothing.
    """
    rewritten = _rewriter().invoke({"question": state["question"]}).strip()
    return {"question": rewritten, "attempts": state.get("attempts", 0) + 1}


def generate_node(state: RAGState) -> dict:
    """Generate with the same compliance prompt as ask.py: cite clauses, refuse when absent."""
    # Answer the user's ORIGINAL question, even though retrieval may have used a rewrite.
    user_message = assemble_prompt(state["original_question"], state["chunks"])
    return {"answer": _generator().invoke({"user_message": user_message})}


def should_continue(state: RAGState) -> str:
    """Conditional edge: generate if relevant or out of attempts, else rewrite."""
    if state.get("relevant") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", should_continue, {"generate": "generate", "rewrite": "rewrite"})
graph.add_edge("rewrite", "retrieve")   # loop back after rewriting
graph.add_edge("generate", END)

app = graph.compile()


def _run(query: str) -> dict:
    """Invoke the graph with the Langfuse callback bound to the current trace.

    Constructed per call for the reason langchain_rag.py documents: the handler
    binds to whatever trace is open, so a module-level one would attach every
    question's spans to whichever trace happened to be first.
    """
    return app.invoke(
        {"question": query, "original_question": query, "attempts": 0},
        config={"callbacks": [CallbackHandler()]},
    )


@observe(name="rag-answer")
def answer_question_graph_with_context(query: str) -> tuple[str, list[dict]]:
    """Drop-in equivalent to ask.answer_question_with_context.

    Scored here rather than in the graph so the refusal and citation_grounding
    scores land on the same trace names the other two implementations use -
    without them the Day 6 false-refusal comparison has nothing to read.
    """
    result = _run(query)
    answer, chunks = result["answer"], result.get("chunks", [])
    # update_current_span, not update_current_trace - the latter is not on the
    # v3 client. Inside @observe this is the trace's root span anyway, so the
    # loop count is visible at the top of the trace where it is useful.
    langfuse.update_current_span(
        metadata={"rag_impl": "langgraph", "attempts": result.get("attempts", 0)},
    )
    _score_answer(answer, chunks)
    return answer, chunks


def answer_question_graph(query: str) -> str:
    return answer_question_graph_with_context(query)[0]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What has to happen when a critical limit at a CCP is exceeded?"
    result = _run(query)
    print(f"Q: {query}\n")
    print(f"Attempts: {result.get('attempts', 0)}")
    print(f"Retrieval question used: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    # Same reason as ask.py: the CLI exits before the SDK's background exporter
    # would have flushed, so an untraced run looks like a bug.
    langfuse.flush()
