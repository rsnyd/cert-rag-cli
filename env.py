"""Load the project's .env before anything reads os.environ.

Credentials used to live only in whichever shell you had exported them into,
which failed in two directions that both stayed quiet:

  - an absent LANGFUSE_PUBLIC_KEY disables the SDK silently, so a run that
    looked normal produced no traces at all
  - a stale one fails per span with a 401 that never stops the run

Both cost a full 39-record eval before this module existed. Keeping the keys in
a gitignored .env means a new terminal, a new day or a fresh clone all start
from the same credentials instead of whatever was last typed by hand.

Real environment variables always win (override=False), so CI and production -
which set them properly - are unaffected by a stray .env on disk.
"""
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parent / ".env"

# Import-time side effect, deliberately: every consumer needs this to have
# happened before its first os.environ read, and an explicit load_env() call
# would just be one more line each module has to remember to put in the right
# place. `import env` is the whole contract. Missing .env is not an error -
# exported variables alone remain a valid way to run.
load_dotenv(ENV_FILE, override=False)
