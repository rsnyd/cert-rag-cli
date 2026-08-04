"""Langfuse tracing setup - import this before any instrumented code runs.

Reads credentials from the environment, loaded from .env by the env module
(see .env.example for the template):

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=http://localhost:3000

If those keys are absent the Langfuse SDK disables itself: every @observe span
becomes a no-op and the app runs exactly as before, just untraced. So importing
this module is always safe, with or without a Langfuse account - which is why
the instrumentation can be written in Week 3 and switched on in Week 4.

If they are present but wrong, the credentials are checked once here rather
than discovered span by span, and TRACING_ENABLED says which of the three
states this process is in: keys absent, keys broken, or tracing live.

The decorated functions live in ask.py, evals/judge.py and evals/run_eval.py.
This module only owns client construction, so there is a single place that tags
every trace with the git commit (release). That is what lets you compare quality
across versions in the Langfuse UI.
"""
import logging
import os
import subprocess
import sys

import requests
from langfuse import get_client

# Loads .env into os.environ. Must come before the reads below, which is why it
# sits with the imports rather than inside a function.
import env  # noqa: F401

DEFAULT_HOST = "https://cloud.langfuse.com"


def _git_release() -> str | None:
    """Short commit SHA, recorded on every trace so runs are comparable across versions."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _auth_ok(host: str, public_key: str, secret_key: str) -> bool | None:
    """Do these credentials work? True/False, or None if the server was unreachable.

    Checked once at startup rather than left to the first span, because the SDK
    exports in a background batch: a bad key surfaces as a per-span 401 on
    stderr that no caller ever sees the return value of, so a long run happily
    finishes having sent nothing. One request here converts that into a single
    message before any work begins.
    """
    try:
        response = requests.get(
            f"{host.rstrip('/')}/api/public/projects",
            auth=(public_key, secret_key),
            timeout=5,
        )
    except requests.RequestException:
        return None
    return response.status_code == 200


_TRACING_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
AUTH_OK: bool | None = None

if _TRACING_ON:
    # The SDK reads release/environment from env vars at client construction, so
    # set sensible defaults before get_client() builds the process-wide
    # singleton. We don't overwrite values the user set explicitly.
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")
    _release = _git_release()
    if _release:
        os.environ.setdefault("LANGFUSE_RELEASE", _release)

    _host = os.environ.get("LANGFUSE_HOST") or DEFAULT_HOST
    AUTH_OK = _auth_ok(_host, os.environ["LANGFUSE_PUBLIC_KEY"],
                       os.environ.get("LANGFUSE_SECRET_KEY", ""))

    if AUTH_OK is not True:
        _reason = ("could not reach the server"
                   if AUTH_OK is None else "the server rejected the credentials")
        _key = os.environ["LANGFUSE_PUBLIC_KEY"]
        print(
            "\n" + "=" * 72 + "\n"
            f"LANGFUSE TRACING DISABLED - {_reason}.\n"
            f"  host: {_host}\n"
            f"  key:  {_key[:14]}...\n"
            "\n"
            "Everything else still runs; only tracing is off. Fix the LANGFUSE_*\n"
            "values in .env (see .env.example) and re-run. The secret key is shown\n"
            "once at creation and stored only as a hash - if it is lost, issue a\n"
            "new pair in the Langfuse UI under Settings -> API Keys.\n"
            + "=" * 72 + "\n",
            file=sys.stderr,
        )
        # Switch the SDK off outright instead of letting it retry every span.
        # Read at construction, so it has to be set before get_client() below.
        os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
        logging.getLogger("langfuse").setLevel(logging.CRITICAL)
else:
    # No credentials: the SDK still runs but every span/update would otherwise
    # log an auth/no-span warning per call. Silence that so untraced runs stay
    # quiet - the @observe spans become harmless no-ops.
    logging.getLogger("langfuse").setLevel(logging.CRITICAL)

# get_client() builds (or returns) the singleton from the LANGFUSE_* env vars.
langfuse = get_client()

# True only when credentials are present AND the server accepted them. Callers
# that exist to produce traces (evals/run_eval.py) refuse to start without it.
TRACING_ENABLED = _TRACING_ON and AUTH_OK is True

__all__ = ["langfuse", "TRACING_ENABLED"]