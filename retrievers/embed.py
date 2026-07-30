"""Query embedding via Voyage.

Voyage's free tier allows 3 requests per minute. embed.py already paces the
ingest side, but query embedding had no pacing and no retry, so a 39-record eval
run burned the minute's quota on its first three records and every call after
that failed outright. Two guards below, because they cover different failures:

  pacing  - never issue calls faster than the tier allows in the first place
  retry   - recover anyway when a limit is hit, since pacing cannot account for
            other processes sharing the same API key

Set VOYAGE_MIN_INTERVAL_SEC=0 to disable pacing once the account has a payment
method and standard rate limits.
"""
import os
import time

import voyageai
from voyageai import error as voyage_error

EMBED_MODEL = "voyage-3-lite"

# 3 RPM means one call every 20s; 21 leaves a margin for clock skew, matching
# the SLEEP_BETWEEN_BATCHES constant embed.py uses on the ingest side.
MIN_INTERVAL_SEC = float(os.getenv("VOYAGE_MIN_INTERVAL_SEC", "21"))
MAX_ATTEMPTS = 5

# Transient by nature: waiting and retrying is the correct response. Auth and
# malformed-request errors are deliberately absent - retrying those just turns a
# clear failure into a slow one.
RETRYABLE = (
    voyage_error.RateLimitError,
    voyage_error.ServerError,
    voyage_error.ServiceUnavailableError,
    voyage_error.APIConnectionError,
)

# One client for the process. Constructing one per call re-read the environment
# and discarded any connection reuse for no benefit.
_client = None
_last_call_at = 0.0


def _voyage() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client()
    return _client


def _wait_for_slot() -> None:
    """Sleep until MIN_INTERVAL_SEC has passed since the previous call."""
    global _last_call_at
    if MIN_INTERVAL_SEC > 0:
        elapsed = time.monotonic() - _last_call_at
        if _last_call_at and elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)
    _last_call_at = time.monotonic()


def embed_query(query: str) -> list[float]:
    """Embed a query, pacing to the rate limit and retrying transient failures.

    Raises the last error if every attempt fails, so a genuinely dead API still
    surfaces as a RAG error in the eval rather than being silently swallowed.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        _wait_for_slot()
        try:
            return _voyage().embed(
                [query], model=EMBED_MODEL, input_type="query"
            ).embeddings[0]
        except RETRYABLE as e:
            last_error = e
            if attempt == MAX_ATTEMPTS - 1:
                break
            # Back off past the full rate-limit window: 21s, 42s, 63s, 84s.
            time.sleep(MIN_INTERVAL_SEC * (attempt + 1) or 2 ** attempt)
    raise last_error
