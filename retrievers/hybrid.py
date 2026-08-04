"""Hybrid retrieval: BM25 keyword + cosine vector, fused via RRF."""
import json
import re
from pathlib import Path

from langfuse import observe
from rank_bm25 import BM25Okapi

from retrievers.vanilla import retrieve as vector_retrieve
from tracing import langfuse

# Anchored to the repo root rather than the cwd. Evals happen to run from the
# repo root, but serve.py can be launched from anywhere, and a relative path
# would make BM25 fail depending only on where the process was started.
CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"

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
        if not CHUNKS_FILE.exists():
            # data/ is gitignored, so a fresh clone has the code but not the
            # corpus. Say which step is missing instead of a bare IO error.
            raise FileNotFoundError(
                f"{CHUNKS_FILE} not found. Hybrid retrieval indexes the same chunks "
                "the vector store was built from - run ingest.py, then chunk.py, "
                "then embed.py."
            )
        _chunks_cache = [json.loads(line) for line in CHUNKS_FILE.open(encoding="utf-8")]
        _bm25_cache = BM25Okapi([_tokenize(_index_text(c)) for c in _chunks_cache])
    return _chunks_cache, _bm25_cache


def _key(result: dict) -> tuple:
    """Identity used to fuse the two result lists.

    The retrievers read different stores - vector from Chroma, BM25 from
    chunks.jsonl - and share no id, because vanilla.retrieve does not return
    one. Chroma holds the chunk text verbatim, so text plus provenance is a
    stable join key across both.

    Deliberately the whole text, not a prefix: two chunks in this corpus share
    their first 100 characters ("INTRODUCTION\\nSpices, Inc. has established,
    documented, and implemented procedure..."), so a prefix key fuses two
    distinct chunks into one and silently drops a result.
    """
    return (result.get("source"), result.get("page"), result["text"])


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
def hybrid_retrieve(query: str, k: int = 5, k_per_retriever: int | None = None,
                    rrf_k: int = 60) -> list[dict]:
    """Run vector + BM25, fuse via Reciprocal Rank Fusion.

    k_per_retriever defaults to max(2k, 20) rather than a fixed 10. ask.py asks
    for TOP_K=14, and a fixed 10 would hand the fusion fewer vector candidates
    than the vanilla retriever sees at that same k - hybrid would then be
    measured against a baseline it was never given the depth to match. Depth is
    nearly free here: both retrievers run off the one embedding call, and BM25
    scores the whole corpus regardless of k.
    """
    if k_per_retriever is None:
        k_per_retriever = max(2 * k, 20)

    vector_results = vector_retrieve(query, k=k_per_retriever)
    bm25_results = bm25_retrieve(query, k=k_per_retriever)

    rrf_scores: dict[tuple, float] = {}
    seen: dict[tuple, dict] = {}

    # Derive rank from list position (1-based). The vector retriever returns
    # results nearest-first but with no "rank" key, so don't rely on one.
    for rank, r in enumerate(vector_results, 1):
        key = _key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rank + rrf_k)
        seen[key] = r

    for rank, r in enumerate(bm25_results, 1):
        key = _key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (rank + rrf_k)
        # Keep the vector copy when both retrievers surfaced the chunk: it
        # carries the cosine distance, which the BM25 record has no analogue for.
        seen.setdefault(key, r)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    # "score" is the fused RRF score, replacing whatever per-retriever score the
    # source record carried. "rank" is the position after fusion.
    results = [
        {**seen[key], "score": score, "rank": rank + 1}
        for rank, (key, score) in enumerate(ranked)
    ]
    langfuse.update_current_span(
        input={"query": query},
        metadata={
            "k": k, "k_per_retriever": k_per_retriever, "rrf_k": rrf_k,
            "n_vector": len(vector_results), "n_bm25": len(bm25_results),
            "n_fused": len(rrf_scores),
            "n_overlap": len(vector_results) + len(bm25_results) - len(rrf_scores),
        },
        output={"top_clauses": [r.get("clause") for r in results]},
    )
    return results
