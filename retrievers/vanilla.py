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
