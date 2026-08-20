"""Day 2: LangChain RAG over the SQF corpus. Ingestion + retrieval.

Loads the clause chunks produced by chunk.py (data/chunks.jsonl) rather than
re-splitting, so clause/page metadata survives for citation. LangChain owns
embedding, storage and retrieval only.
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from rich.progress import track

# Same contract as embed.py: VOYAGE_API_KEY comes from .env, not from whatever
# the current shell happens to have exported.
import env  # noqa: F401
from retrievers.embed import EMBED_MODEL, embed_query, paced_call

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


def get_retriever(k: int = 5):
    return _store().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_index()
    else:
        retriever = get_retriever()
        results = retriever.invoke("How often must internal audits be conducted?")
        for i, doc in enumerate(results, 1):
            clause = doc.metadata.get("clause", "-")
            print(f"\n--- Result {i} ({doc.metadata.get('source')} clause {clause}) ---")
            print(doc.page_content[:200])