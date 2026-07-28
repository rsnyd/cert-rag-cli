"""Langfuse tracing setup - import this before any instrumented code runs.

Reads credentials from the environment (same pattern as the Anthropic and
Voyage clients elsewhere in this repo):

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000

If those keys are absent the Langfuse SDK disables itself: every @observe span
becomes a no-op and the app runs exactly as before, just untraced. So importing
this module is always safe, with or without a Langfuse account - which is why
the instrumentation can be written in Week 3 and switched on in Week 4.

The decorated functions live in ask.py, evals/judge.py and evals/run_eval.py.
This module only owns client construction, so there is a single place that tags
every trace with the git commit (release). That is what lets you compare quality
across versions in the Langfuse UI.
"""
import logging
import os
import subprocess

from langfuse import get_client


def _git_release() -> str | None:
    """Short commit SHA, recorded on every trace so runs are comparable across versions."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


_TRACING_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))

if _TRACING_ON:
    # The SDK reads release/environment from env vars at client construction, so
    # set sensible defaults before get_client() builds the process-wide
    # singleton. We don't overwrite values the user set explicitly.
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    _release = _git_release()
    if _release:
        os.environ.setdefault("LANGFUSE_RELEASE", _release)
else:
    # No credentials: the SDK still runs but every span/update would otherwise
    # log an auth/no-span warning per call. Silence that so untraced runs stay
    # quiet - the @observe spans become harmless no-ops.
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)

# get_client() builds (or returns) the singleton from the LANGFUSE_* env vars.
langfuse = get_client()

__all__ = ["langfuse"]