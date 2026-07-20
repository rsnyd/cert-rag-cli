"""Day 12: Ask a question. Retrieve context. Generate an answer."""
import os
import sys

from anthropic import Anthropic

from retrievers.vanilla import retrieve

TOP_K = 5
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


def assemble_prompt(query: str, chunks: list[dict]) -> str:
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


def answer_with_sources(query: str) -> dict:
    """Retrieve, generate, and return the answer alongside the cited sources.

    Returns {"answer": str, "sources": [{source, clause, clause_title, page}]}.
    The source list is the retrieved chunks (minus their full text), so a caller
    such as a web UI can render citations without re-running retrieval.
    """
    chunks = retrieve(query, k=TOP_K)
    prompt = assemble_prompt(query, chunks)
    anthropic = Anthropic()
    response = anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    sources = [
        {
            "source": c["source"],
            "clause": c.get("clause"),
            "clause_title": c.get("clause_title"),
            "page": c.get("page"),
        }
        for c in chunks
    ]
    return {"answer": response.content[0].text, "sources": sources}


def answer_question(query: str) -> str:
    return answer_with_sources(query)["answer"]


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python ask.py "your question"')
        sys.exit(1)
    print(answer_question(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()