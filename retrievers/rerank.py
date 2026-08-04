"""Hybrid retrieval + Voyage rerank.

Reranking is the one strategy whose cost is a whole candidate set rather than a
single query, so the free tier's token ceiling binds here in a way it does not
elsewhere. Voyage caps an unbilled account at 3 RPM *and 10K TPM*, and a rerank
request pays for every candidate it scores. Sending 56 candidates raised a
RateLimitError that no amount of backoff could clear - a request larger than
the per-minute budget can never succeed, so it retried for 250s and then failed.

Candidates are therefore trimmed to a token budget rather than a fixed count.
Chunks in this corpus run from ~349 characters at the median to 2400 at p95, a
7x spread, so any fixed count is either wasteful on small chunks or over the
ceiling on large ones: 20 large chunks is ~12K tokens, already past the limit.
"""
import os

from langfuse import observe

# Not a fresh voyageai.Client(): rerank spends a request from the same account
# rate limit the query embedding does, so it goes through the shared gate in
# embed.py. See paced_call's docstring for why that matters at eval scale.
from retrievers.embed import paced_call
from retrievers.hybrid import hybrid_retrieve
from tracing import langfuse

RERANK_MODEL = "rerank-2.5"

# Sized for sustained eval throughput, not for one question. Pacing applies per
# Voyage call and rerank spends two per question (embedding + rerank), so the
# floor is 2 x 21s = 42s, or ~1.43 questions/minute against a 10K TPM ceiling.
#
# Measured over the 39 golden questions, taking BM25's top 56 as the candidate
# set:
#
#   budget  kept (median/min)  questions under k=14  sustained TPM
#   6000        18 / 13               3 / 39             8,275
#   7000        21 / 15               0 / 39             9,679
#   8000        24 / 18               0 / 39            11,102   over ceiling
#
# 7000 never truncates below k and still fits, but only by 3%, and that margin
# rests on the estimate below. This corpus is dense with clause numbers like
# 2.5.5.1, which tokenize far worse than prose, so the real count runs above a
# 4-chars-per-token guess and 7000 would likely breach in practice. 6000 costs
# one excerpt on 3 of 39 questions (13 rather than 14) and keeps real headroom.
#
# This is the knob that buys rerank depth, not k_pre_rerank on its own. Raise it
# once the account has a payment method and the 3 RPM / 10K TPM cap is gone.
RERANK_TOKEN_BUDGET = int(os.getenv("VOYAGE_RERANK_TOKEN_BUDGET", "6000"))

# Rough English ratio. Deliberately an estimate: the point is to stay under a
# ceiling, and paying an extra API call to count tokens exactly would spend the
# very budget being measured.
CHARS_PER_TOKEN = 4


def _fit_token_budget(candidates: list[dict], budget: int) -> tuple[list[dict], int]:
    """Take candidates in rank order until the token budget is spent.

    Always keeps at least one, so a single oversized chunk degrades to a
    one-document rerank instead of an empty request Voyage would reject.
    """
    kept: list[dict] = []
    used = 0
    for c in candidates:
        cost = len(c["text"]) // CHARS_PER_TOKEN + 1
        if kept and used + cost > budget:
            break
        kept.append(c)
        used += cost
    return kept, used


@observe(name="rerank-model", as_type="retriever")
def _rerank_call(query: str, documents: list[str], top_k: int):
    """The Voyage call on its own span.

    Separated from the enclosing "rerank" span so the reranker's latency is
    readable against the retrieval that fed it - otherwise a slow question looks
    equally attributable to Chroma, BM25 or Voyage.
    """
    response = paced_call("rerank", query=query, documents=documents,
                          model=RERANK_MODEL, top_k=top_k)
    langfuse.update_current_span(
        input={"query": query, "n_documents": len(documents)},
        metadata={"model": RERANK_MODEL, "top_k": top_k},
        output={"scores": [round(r.relevance_score, 4) for r in response.results]},
    )
    return response


@observe(name="rerank", as_type="retriever")
def rerank_retrieve(query: str, k: int = 5,
                    k_pre_rerank: int | None = None) -> list[dict]:
    """Get candidates from hybrid, rerank with Voyage, return the top k.

    k_pre_rerank is the ceiling on how many candidates are *considered*; on the
    free tier RERANK_TOKEN_BUDGET is what actually decides the depth, since a
    request over 10K TPM fails no matter how few candidates it names. It
    defaults to max(4k, 40) so that a billed account with the budget raised gets
    real headroom - reranking 14 out of 20 has only six candidates to discard,
    which is rarely enough to change the answer.
    """
    if k_pre_rerank is None:
        k_pre_rerank = max(4 * k, 40)

    candidates = hybrid_retrieve(query, k=k_pre_rerank)
    if not candidates:
        # Voyage rejects an empty document list, and there is nothing to rank.
        return []

    n_retrieved = len(candidates)
    candidates, est_tokens = _fit_token_budget(candidates, RERANK_TOKEN_BUDGET)
    response = _rerank_call(query, [c["text"] for c in candidates], top_k=k)

    # Voyage returns indices into the input list along with relevance scores.
    # Carry the clause metadata across or citation breaks at the last hop.
    results = [
        {
            "text": candidates[r.index]["text"],
            "source": candidates[r.index]["source"],
            "clause": candidates[r.index].get("clause"),
            "clause_title": candidates[r.index].get("clause_title"),
            "page": candidates[r.index].get("page"),
            "doc_type": candidates[r.index].get("doc_type"),
            "rank": rank + 1,
            "score": r.relevance_score,
            "original_rank": candidates[r.index].get("rank"),
        }
        for rank, r in enumerate(response.results)
    ]
    langfuse.update_current_span(
        input={"query": query, "n_candidates": len(candidates)},
        metadata={
            "k": k, "k_pre_rerank": k_pre_rerank, "model": RERANK_MODEL,
            "n_retrieved": n_retrieved, "n_reranked": len(candidates),
            "est_tokens": est_tokens, "token_budget": RERANK_TOKEN_BUDGET,
            # True when the budget, not k_pre_rerank, set the depth. Worth
            # filtering on: it also means fewer than k excerpts reached the
            # model if the budget cut below k.
            "budget_trimmed": len(candidates) < n_retrieved,
            "returned_fewer_than_k": len(results) < k,
        },
        output={
            "top_clauses": [r["clause"] for r in results],
            "rank_changes": [(r["original_rank"], r["rank"]) for r in results],
        },
    )
    return results
