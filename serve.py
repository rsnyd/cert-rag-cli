"""HTTP service exposing the SQF RAG pipeline for the Drupal admin UI.

A thin wrapper around ask.answer_question_with_context - it adds no retrieval or
generation logic of its own, so the CLI and the service always answer the same
way. Run locally with:

    uv run uvicorn serve:app --reload --port 8000

    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "How often are internal audits required?"}' | jq

Auth: if the RAG_API_KEY environment variable is set, every /ask request must
send a matching `X-API-Key` header. If it is unset, auth is disabled (fine for
local dev, not for the live site - set it in production).
"""
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ask import answer_question_with_context

app = FastAPI(title="cert-rag-cli", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    clause: str | None = None
    clause_title: str | None = None
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _check_auth(x_api_key: str | None) -> None:
    expected = os.environ.get("RAG_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _sources(chunks: list[dict]) -> list[Source]:
    """Retrieved chunks as citable sources, in rank order, deduplicated.

    Deduplicated because a long clause is sub-split into several chunks that
    share a clause and page, and retrieval routinely returns more than one of
    them - a UI listing the same citation repeatedly looks like a bug. Text is
    deliberately not returned: the answer already quotes what it relies on, and
    the excerpt bodies are large enough to dominate the response.
    """
    seen, out = set(), []
    for c in chunks:
        key = (c.get("source"), c.get("clause"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(Source(source=c["source"], clause=c.get("clause"),
                          clause_title=c.get("clause_title"), page=c.get("page")))
    return out


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)) -> AskResponse:
    _check_auth(x_api_key)
    answer, chunks = answer_question_with_context(req.question)
    return AskResponse(answer=answer, sources=_sources(chunks))
