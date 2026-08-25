"""Day 2: LangChain RAG over the SQF corpus. Ingestion, retrieval, generation.

Loads the clause chunks produced by chunk.py (data/chunks.jsonl) rather than
re-splitting, so clause/page metadata survives for citation. LangChain owns
embedding, storage, retrieval and the generation chain.

The prompt, the model and k come from ask.py rather than being restated here.
The point of this file is to find out whether LCEL is worth adopting, and that
comparison only means something if the two pipelines differ in their plumbing
and nothing else. A prompt that drifts by a blank line or a k that differs from
the swept value would show up as a quality difference that has nothing to do
with LangChain.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langfuse import observe
from langfuse.langchain import CallbackHandler
from rich.progress import track

# Same contract as embed.py: VOYAGE_API_KEY comes from .env, not from whatever
# the current shell happens to have exported.
import env  # noqa: F401
# _score_answer is private to ask.py, and imported anyway: the two reference-free
# scores are what make a LangChain answer comparable to a vanilla one in the
# Langfuse UI, and a second copy of that logic here would be the same drift risk
# as a second copy of the prompt.
from ask import LLM_MODEL, SYSTEM_PROMPT, TOP_K, _score_answer, assemble_prompt
from retrievers.embed import EMBED_MODEL, embed_query, paced_call
from tracing import langfuse

CHUNKS_FILE = Path("data/chunks.jsonl")
PERSIST_DIR = ".chroma_langchain"   # separate dir so we don't clobber the raw-API index
COLLECTION = "sqf_docs_lc"

# Voyage's free tier is 3 RPM / 10K TPM. langchain_voyageai batches by token
# budget and fires the batches back to back, so a plain Chroma.from_documents()
# over 971 chunks trips the limit on the first request. Keep embed.py's batch
# size for the token budget; the request spacing comes from paced_call.
BATCH = 8

# Chroma metadata values must be str/int/float/bool - never None (same constraint
# as embed.py). Drop None keys so preamble/DOCX chunks don't break the load.
_META_FIELDS = ("source", "doc_type", "edition", "module", "clause", "clause_title", "page")


def _to_document(rec: dict) -> Document:
    metadata = {k: rec[k] for k in _META_FIELDS if rec.get(k) is not None}
    return Document(page_content=rec["text"], metadata=metadata)


class PacedVoyageEmbeddings(Embeddings):
    """Voyage embeddings behind the pace-and-retry gate in retrievers/embed.py.

    langchain_voyageai's VoyageAIEmbeddings builds its own voyageai.Client, and
    that SDK defaults to max_retries=0 - so the first 429 propagates straight
    out of retriever.invoke(). Its pydantic model sets extra="forbid" and has no
    retry field, so there is nowhere to pass a configured client in. Hence a
    plain Embeddings implementation instead of a subclass.

    Going through paced_call also puts both halves of this script on the same
    process-wide 21s spacing as the raw-API path, which matters here: without
    it, a query issued right after a build lands inside the same 3 RPM window as
    the final ingest batch and is rejected on arrival.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = paced_call("embed", texts=texts, model=EMBED_MODEL,
                              input_type="document")
        return response.embeddings

    def embed_query(self, text: str) -> list[float]:
        # The shared helper, so a LangChain query produces the same Langfuse
        # embedding span as a vanilla one.
        return embed_query(text)


def _store() -> Chroma:
    embeddings = PacedVoyageEmbeddings()
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def build_index():
    records = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
    print(f"Loaded {len(records)} clause chunks")

    vectorstore = _store()

    # Chunk ids are stable and langchain_chroma upserts, so a build cut short by
    # a rate limit or a Ctrl-C resumes where it stopped instead of re-embedding
    # (and re-waiting for) everything that already landed.
    indexed = set(vectorstore.get(include=[])["ids"])
    pending = [r for r in records if r["id"] not in indexed]
    if indexed:
        print(f"{len(indexed)} already indexed, embedding the remaining {len(pending)}")

    # No sleep in the loop: PacedVoyageEmbeddings already waits for its slot
    # before every request, including the one after the last batch. That last
    # interval is the one a manual "sleep between batches" leaves out, and it is
    # exactly the gap a query issued right after the build falls into.
    batches = [pending[i:i + BATCH] for i in range(0, len(pending), BATCH)]
    for batch in track(batches, description="Embedding"):
        vectorstore.add_documents(
            documents=[_to_document(r) for r in batch],
            ids=[r["id"] for r in batch],
        )

    print(f"Indexed {vectorstore._collection.count()} chunks in {PERSIST_DIR}")
    return vectorstore


@lru_cache(maxsize=None)
def get_retriever(k: int = TOP_K):
    """Cached per k: _store() opens a fresh Chroma client and a fresh embeddings
    object every call, which is startup work, not per-question work. It matters
    for langgraph_rag.py, whose retrieve node runs once per loop iteration.

    k defaults to TOP_K for the same reason the prompt is imported rather than
    restated - a different k here would show up in the Day 6 three-way
    comparison as a framework effect.
    """
    return _store().as_retriever(search_kwargs={"k": k})


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# SYSTEM_PROMPT arrives as a SystemMessage rather than a ("system", ...) tuple.
# A tuple is parsed as an f-string-style template, so the day someone puts a
# brace in ask.py's prompt this would fail with a missing-variable error in a
# file that has nothing to do with the edit. A SystemMessage is passed through
# verbatim. The human turn is templated, but only the template string is parsed
# - clause text substituted into {user_message} is data, braces and all.
PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    ("human", "{user_message}"),
])


def _doc_to_chunk(doc: Document) -> dict:
    """Convert a LangChain Document to the chunk dict shape the eval expects."""
    m = doc.metadata
    return {
        "text": doc.page_content,
        "source": m.get("source"),
        "clause": m.get("clause"),
        "clause_title": m.get("clause_title"),
        "page": m.get("page"),
        "doc_type": m.get("doc_type"),
    }


def _docs_to_chunks(docs: list[Document]) -> list[dict]:
    return [_doc_to_chunk(d) for d in docs]


@lru_cache(maxsize=None)
def build_rag_chain(k: int = TOP_K):
    """Retrieval + generation as one LCEL chain, returning the answer AND its context.

    Cached per k because every call to _store() builds a fresh Chroma client and
    a fresh embeddings object, and init_chat_model resolves and constructs a
    provider client. None of that is per-question work.

    The chain carries `chunks` alongside the answer rather than retrieving twice.
    That is not just tidiness: under the pacing gate a second retrieval costs a
    real 21 seconds, so a retrieve-then-answer helper that ignored the chain's
    own retrieval would double the wall-clock of a 39-record eval.

    The user turn is built by ask.assemble_prompt, so the model sees exactly the
    same bytes it sees on the vanilla path - the excerpt headers are what it
    cites, and a formatter reimplemented here would drift from that one.
    """
    return (
        RunnableParallel({
            "chunks": get_retriever(k=k) | _docs_to_chunks,
            "question": RunnablePassthrough(),
        })
        | RunnablePassthrough.assign(
            user_message=lambda x: assemble_prompt(x["question"], x["chunks"]),
        )
        | RunnablePassthrough.assign(
            answer=PROMPT | _chat_model() | StrOutputParser(),
        )
    )


@lru_cache(maxsize=None)
def _chat_model():
    """The same model ask.py calls, reached through LangChain instead of the SDK.

    model_provider is explicit: init_chat_model would infer "anthropic" from the
    claude- prefix, but the inference is a lookup table, and being wrong about it
    surfaces as a missing-package error rather than as a wrong provider.

    No temperature. ask.py does not set one, so setting 0 here would make the
    LangChain path deterministic and the vanilla path not - a difference the
    strategy comparison would report as a LangChain effect. (It is also worth
    knowing that temperature is rejected outright on Opus 5 / Sonnet 5 and the
    4.7+ family, so a hardcoded one here is a migration hazard as well.)
    """
    return init_chat_model(LLM_MODEL, model_provider="anthropic", max_tokens=1024)


@observe(name="rag-answer")
def answer_question_lc_with_context(query: str, k: int = TOP_K) -> tuple[str, list[dict]]:
    """Drop-in equivalent to ask.answer_question_with_context.

    The Langfuse callback is constructed per call by design - it binds to the
    trace @observe just opened, so a module-level handler would attach every
    question's spans to whichever trace happened to be first.
    """
    result = build_rag_chain(k).invoke(
        query, config={"callbacks": [CallbackHandler()]},
    )
    answer, chunks = result["answer"], result["chunks"]
    _score_answer(answer, chunks)
    return answer, chunks


def answer_question_lc(query: str) -> str:
    """Answer text only, for the CLI."""
    return answer_question_lc_with_context(query)[0]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "ask":
        print(answer_question_lc(" ".join(sys.argv[2:])))
        # Same reason as ask.py: the CLI exits before the SDK's background
        # exporter would have flushed, so an untraced run looks like a bug.
        langfuse.flush()
    else:
        retriever = get_retriever()
        results = retriever.invoke("How often must internal audits be conducted?")
        for i, doc in enumerate(results, 1):
            clause = doc.metadata.get("clause", "-")
            print(f"\n--- Result {i} ({doc.metadata.get('source')} clause {clause}) ---")
            print(doc.page_content[:200])
