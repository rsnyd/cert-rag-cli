"""Query embedding via Voyage, with rate-limit pacing and backoff.

Lives in the retrievers package (not ask.py) so every retrieval strategy can
share it without importing ask.py, which would be a circular import - ask.py
imports a retriever at module load.

Voyage's free tier allows 3 requests per minute. Two guards below, because they
cover different failures:

  pacing  - never issue calls faster than the tier allows in the first place
  retry   - recover anyway when a limit is hit, since pacing cannot account for
            other processes sharing the same API key

Pacing is the one that carries a full eval. A 39-record run paces to about 2.6
calls/min and finishes in ~15 minutes with no rejected call. Drop it and the
run bursts straight past the limit, then every call pays an unpredictable
backoff instead of a predictable interval.

Set VOYAGE_MIN_INTERVAL_SEC=0 to disable pacing once the account has a payment
method and standard rate limits.
"""
import os
import sys
import time

import voyageai
from langfuse import observe
from voyageai import error as voyage_error

from tracing import langfuse

EMBED_MODEL = "voyage-3-lite"

# 3 RPM means one call every 20s; 21 leaves a margin for clock skew, matching
# the SLEEP_BETWEEN_BATCHES constant embed.py uses on the ingest side.
MIN_INTERVAL_SEC = float(os.getenv("VOYAGE_MIN_INTERVAL_SEC", "21"))

# Transient by nature: waiting and retrying is the correct response. Auth and
# malformed-request errors are deliberately absent - retrying those just turns a
# clear failure into a slow one. Matching on the exception type rather than on
# substrings of its message, so a server error is not misread as throttling
# because its text happened to contain a number.
RETRYABLE = (
    voyage_error.RateLimitError,
    voyage_error.ServerError,
    voyage_error.ServiceUnavailableError,
    voyage_error.APIConnectionError,
)

# One client for the process. Constructing one per call re-read the environment
# and discarded any connection reuse for no benefit.
_client = None
# None rather than 0.0 means "no call yet". monotonic()'s zero point is
# undefined, so a real reading can legitimately be 0.0 and a truthiness test
# would silently skip the first interval.
_last_call_at: float | None = None


def _voyage() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client()
    return _client


def _wait_for_slot() -> None:
    """Sleep until MIN_INTERVAL_SEC has passed since the previous call."""
    global _last_call_at
    if MIN_INTERVAL_SEC > 0 and _last_call_at is not None:
        elapsed = time.monotonic() - _last_call_at
        if elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)
    _last_call_at = time.monotonic()


def paced_call(operation: str, *, max_retries: int = 6, **kwargs):
    """Run one Voyage API call under the process-wide pace and retry policy.

    Every Voyage endpoint bills against the same account rate limit, so they all
    queue behind this one gate. That matters for the rerank strategy, which
    spends two requests per question - one embedding, one rerank. Pacing only
    the embedding would let the rerank half slip past the limit unmetered and
    put the run straight back into the 429s pacing exists to avoid.

    `operation` names the method on the client: "embed", "rerank".

    Retries back off exponentially (10s, 20s, 40s, then capped at 60s) so a call
    that arrives mid-window waits out the whole per-minute bucket rather than
    hammering it. Pacing still applies between attempts, so the two compose to
    max(interval, backoff): the first couple of retries land on the 21s pacing
    floor and the later ones dominate it.

    Raises the last error if every attempt fails, so a genuinely dead API still
    surfaces as a RAG error in the eval rather than being silently swallowed.
    """
    for attempt in range(max_retries + 1):
        _wait_for_slot()
        try:
            return getattr(_voyage(), operation)(**kwargs)
        except RETRYABLE as err:
            if attempt == max_retries:
                raise
            delay = min(60, 10 * (2 ** attempt))
            print(f"    {type(err).__name__} from Voyage {operation}, retrying in "
                  f"{delay}s (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
            time.sleep(delay)


@observe(name="embed-query", as_type="embedding",
         capture_input=False, capture_output=False)
def embed_query(query: str, max_retries: int = 6) -> list[float]:
    """Embed a query, paced and retried by paced_call.

    Traced as an "embedding" observation. We suppress the default input/output
    capture and set them by hand: logging the query text is useful, but the raw
    float vector is noise in the UI, so we record only its dimensionality.
    """
    langfuse.update_current_generation(model=EMBED_MODEL, input=query)
    response = paced_call("embed", max_retries=max_retries,
                          texts=[query], model=EMBED_MODEL, input_type="query")
    embedding = response.embeddings[0]
    langfuse.update_current_generation(metadata={"dimensions": len(embedding)})
    return embedding
