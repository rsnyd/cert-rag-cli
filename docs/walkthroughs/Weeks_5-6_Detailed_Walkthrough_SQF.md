# Weeks 5-6 Detailed Walkthrough: LLM Engineering (Frameworks + Adaptation)

**Working tutorial, not just a plan. Aim: by end of Week 6 you can build the same
SQF compliance RAG in raw API and in LangChain, you understand LCEL and LangGraph
well enough to discuss them in an interview, and you've fine-tuned and published
one small LoRA adapter on your own brand-voice data.**

**The theme shift**: Weeks 1-4 were "build it yourself so you understand it."
Weeks 5-6 are "now learn the tools the industry actually uses, with the
understanding you earned." You'll see exactly what LangChain automates and exactly
what it costs you. That contrast - "I can do it both ways and here's the tradeoff"

- is one of the strongest things you can say in an SA interview.

---

## Before You Start: Prerequisites Check (15 minutes)

Coming into Week 5 you should have, from Weeks 1-4:

- [ ] `~/projects/cert-rag-cli` repo with: vanilla/hybrid/rerank retrievers over the SQF corpus, clause-aware chunking with clause/page metadata, a 30-question SQF golden set plus refusal probes, the five-axis judge (factual, complete, relevant, citation, grounding), deterministic clause-hit and refusal metrics, Langfuse instrumentation
- [ ] `EVAL_REPORT.md` with the three-strategy comparison
- [ ] Anthropic + Voyage + OpenAI keys in the project's `.env` (loaded by `env.py`)
- [ ] Langfuse running locally (or Langfuse Cloud configured)
- [ ] WSL2 + uv + Python 3.12

One thing to confirm before Day 1: **check whether ROBDEV2 has an NVIDIA GPU**.
This matters for Week 6.

```bash
# In WSL2
nvidia-smi
```

If that prints a GPU table, you have CUDA available and can fine-tune locally. If
it errors with "command not found" or "no devices," that's fine - Week 6 uses
Google Colab's free T4 GPU instead, and the walkthrough assumes the Colab path by
default since it works for everyone. You do not need a local GPU to complete this
plan.

### Install the new dependencies

LangChain 1.x split into separate provider packages. Install the current set:

```bash
cd ~/projects/cert-rag-cli
uv add langchain langchain-anthropic langchain-voyageai langchain-chroma langchain-community langgraph
```

A note on versions: LangChain moves fast and makes breaking changes between minor
versions. **Pin your versions** so your code doesn't break under you mid-week:

```bash
uv pip freeze | grep -i langchain
# Note the versions, they're now pinned in your uv.lock
```

If a tutorial you find online uses `LLMChain`, `SequentialChain`,
`ConversationChain`, or `RetrievalQA`, it's pre-1.0 and outdated. The current API
is LCEL (the pipe `|` operator) for chains and LangGraph for agents. Ignore old
tutorials.

---

# WEEK 5: LangChain and LangGraph

**Goal**: Reimplement your SQF RAG using LangChain's LCEL, then build a small
agentic-RAG variant in LangGraph. End the week able to articulate when a framework
helps and when it gets in the way, and with all three implementations scored on
the identical SQF golden set including the citation and refusal metrics.

---

## Day 1 (Mon): What LangChain Is and Why It Exists

No coding today beyond a "hello chain." Concepts and orientation.

### Concept: The problem LangChain solves

You've spent four weeks writing raw API calls. You wrote your own retrieval loop,
your own prompt assembly, your own retry handling. That was the right way to
learn. But you probably noticed you were rewriting the same plumbing each time:
load chunks, embed query, retrieve, format a prompt, call the model, parse the
response.

LangChain's pitch is: that plumbing is the same across every LLM app, so let's
standardize it. It gives you:

- **A unified model interface.** Swap Claude for GPT for Gemini by changing one
  line. Your code doesn't care which provider it's talking to.
- **Composable components.** Prompts, models, retrievers, parsers, and tools all
  implement the same `Runnable` interface, so they snap together.
- **An ecosystem.** 600+ integrations. Every vector DB, every model provider,
  every document loader has a LangChain class already written.

The cost: it's another abstraction layer on top of an already-abstract API. When
things break, you're debugging through LangChain's internals instead of your own
code. The criticism is real and you should hold both views at once: useful
ecosystem, leaky abstraction. You'll hit a concrete example of the leak on Day 2,
where LangChain's default chunker is wrong for this corpus.

### Concept: LCEL (LangChain Expression Language)

The heart of modern LangChain. Think UNIX pipes. In bash you write
`cat file | grep pattern | sort`. In LCEL you write:

```python
chain = prompt | model | output_parser
```

Each component takes input, transforms it, passes output to the next. Every
component implements `Runnable`, which means it has `.invoke()` (run once),
`.stream()` (stream tokens), and `.batch()` (run many in parallel). You get
streaming, async, and batching for free without writing any of it.

This replaces the old `LLMChain` and `SequentialChain` classes, which are
deprecated. If you internalize one thing this week: **LCEL is the pipe operator,
and it's the current way to build chains.**

### Concept: The four sibling products

- **LangChain**: the core toolkit (chains, retrievers, simple agents). What you'll
  use most.
- **LangGraph**: graph-based orchestration for stateful, multi-step agents with
  loops and branches. You'll use this Day 5 and heavily in Weeks 7-9.
- **LangSmith**: observability and evals (the LangChain team's equivalent to
  Langfuse). You already know Langfuse, so you'll mostly skip LangSmith, but know
  it exists.
- **LangServe**: turns any chain into a REST API. Not needed for this plan.

### Project: One hello chain

Create a scratch file `langchain_hello.py`:

```python
"""Day 1: the smallest possible LCEL chain."""
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# init_chat_model infers the provider from the model string
model = init_chat_model("claude-sonnet-4-6", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an SQF food-safety expert. Be concise. Use plain hyphens, not em dashes."),
    ("human", "{question}"),
])

# The chain: prompt -> model -> string parser
chain = prompt | model | StrOutputParser()

# Run it
answer = chain.invoke({"question": "In two sentences, what is the role of the SQF practitioner?"})
print(answer)
```

Run it:

```bash
uv run python langchain_hello.py
```

Compare this to your Week 1 `hello.py`. Same result, fewer lines, but a layer of
abstraction between you and the API. Notice what you gained (provider-agnostic,
composable) and what you lost (you can no longer see the raw request).

### Reading (1 hour)

- LangChain introduction / quickstart: https://python.langchain.com/docs/introduction/
- LCEL conceptual guide: https://python.langchain.com/docs/concepts/lcel/
- Skim the "Why LangChain?" framing in any current (2026) overview to internalize
  the standardization argument

Commit:

```bash
git add langchain_hello.py
git commit -m "Day 1 (wk5): first LCEL chain, hello world"
```

---

## Day 2 (Tue): Reimplement Retrieval in LangChain

### Concept: LangChain's RAG components

LangChain has a class for every step of the RAG pipeline you built by hand:

- **Document loaders** (`WebBaseLoader`, `TextLoader`, `JSONLoader`) - read source content
- **Text splitters** (`RecursiveCharacterTextSplitter`) - your `chunk.py`, but boundary-aware
- **Embeddings** (`VoyageAIEmbeddings`) - your `embed.py`
- **Vector stores** (`Chroma`) - your Chroma usage
- **Retrievers** (`vectorstore.as_retriever()`) - your `retrieve()`

Here is the first concrete instance of the leaky-abstraction point. LangChain's
`RecursiveCharacterTextSplitter` is a genuine upgrade over naive
character-counting for prose: it splits on natural boundaries (paragraph, then
sentence, then word) before falling back to raw character count. For the Drupal
prose corpus it would have been a strict improvement.

For this corpus it is wrong. Your Week 2 chunker splits on SQF clause boundaries
and attaches the clause number, title and page to each chunk. That metadata is
what every downstream citation depends on, and it is exactly what a generic
recursive splitter would shred: it would cut mid-clause, merge two requirements
into one chunk, and carry no clause number at all. So you are not going to let
LangChain re-chunk. You are going to feed LangChain the clause chunks you already
produced in `data/chunks.jsonl`, and let it own only embedding, storage and
retrieval.

That decision - "the framework's default is wrong for my domain, so I keep my own
chunking and let the framework do the rest" - is a stronger interview story than
"I used the built-in splitter." It shows you can use a framework without being
captured by its defaults.

### Project: `langchain_rag.py` (ingestion + retrieval)

Build a LangChain version of your SQF RAG that reuses your clause chunks, so the
corpus, the chunking and the clause metadata are identical to the raw-API build
and the only variable is the framework.

The same Voyage pacing that carries the raw-API build applies here, and this is
the part of the day that will actually cost you time if you get it wrong.
`langchain_voyageai` batches by token budget and fires the batches back to back,
so a plain `Chroma.from_documents()` over the whole corpus trips the free tier on
the first request.

The obvious fix - `VoyageAIEmbeddings(model=..., batch_size=8)` plus a
`time.sleep(21)` between batches - paces the **build** and leaves the **query**
side ungated, which is a bug you will not see until the demo. `retriever.invoke()`
embeds the query through that same `VoyageAIEmbeddings` object, and
`langchain_voyageai` constructs its own `voyageai.Client` with `max_retries=0`,
so the first 429 propagates straight out of the retriever. You cannot configure
around it either: its pydantic model sets `extra="forbid"` and has no retry
field, so there is nowhere to hand it a client you built.

So implement `Embeddings` directly and route both halves through the `paced_call`
gate you already wrote in `retrievers/embed.py` (Week 4 Day 9). That gate is
process-wide, which is the point - a query issued right after a build otherwise
lands inside the same 3 RPM window as the final ingest batch and is rejected on
arrival. Note also what disappears: there is no `time.sleep` in the build loop
any more, because `paced_call` waits for its slot *before* every request,
including the one after the last batch. That final interval is exactly the gap a
manual sleep-between-batches leaves open.

While you are here, stop redefining the prompt, the model and `k`. Import
`SYSTEM_PROMPT`, `LLM_MODEL` and `TOP_K` from `ask.py`. The entire point of this
file is to find out whether LCEL is worth adopting, and that comparison only
means something if the two pipelines differ in their plumbing and nothing else -
a prompt that drifts by a blank line, or a `k` that differs from the swept value,
shows up as a quality difference that has nothing to do with LangChain.

```python
"""Day 2: LangChain RAG over the SQF corpus. Ingestion + retrieval.

Loads the clause chunks produced by chunk.py (data/chunks.jsonl) rather than
re-splitting, so clause/page metadata survives for citation. LangChain owns
embedding, storage and retrieval.
"""
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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
```

Build the index, then test retrieval:

```bash
uv run python langchain_rag.py build
uv run python langchain_rag.py
```

You should see five retrieved chunks, each carrying a clause number in its
metadata. That clause number surviving into the retriever is the whole reason you
skipped the recursive splitter, and it is what makes the citation and clause-hit
metrics work on this implementation in two days.

Commit:

```bash
git add langchain_rag.py
git commit -m "Day 2 (wk5): LangChain RAG over SQF clause chunks, retrieval"
```

---

## Day 3 (Wed): Complete the LangChain RAG Chain

### Concept: RunnableParallel and RunnablePassthrough

The classic LCEL RAG pattern needs to do two things at once: retrieve context AND
pass the original question through to the prompt. That's what `RunnableParallel`
and `RunnablePassthrough` are for.

```python
retrieval = RunnableParallel({
    "context": retriever,            # runs the retriever on the input
    "question": RunnablePassthrough() # passes the input through unchanged
})
chain = retrieval | prompt | model | output_parser
```

When you `chain.invoke("some question")`, the question goes to both the retriever
(producing `context`) and through the passthrough (producing `question`), then
both feed the prompt. This parallel-then-combine shape is the canonical LCEL RAG
idiom. Know it for interviews.

### Concept: matching the raw-API behavior exactly

For the eval comparison to be apples to apples, the LangChain version has to
behave like `ask.py`, not just answer the same questions. That means two things:

- The prompt has to carry the clause and page into the context so the model can
  cite, and use the same refusal-first compliance system prompt.
- The answer function has to return the retrieved chunks alongside the answer,
  because your Week 3 eval harness scores citation and grounding against the
  context the model saw and computes clause-hit from it.

The strongest way to guarantee both is not to reimplement them. **Import
`SYSTEM_PROMPT`, `LLM_MODEL`, `TOP_K` and `assemble_prompt` from `ask.py`.** A
`format_docs` written here would drift from `assemble_prompt` the first time
either changes, and then your framework comparison is measuring a prompt
difference you did not intend. This is the single-source-of-truth rule in
`CLAUDE.md`, and Week 5 is exactly the week it earns its keep.

Import `_score_answer` too, private underscore and all. The two reference-free
scores are what make a LangChain answer comparable to a vanilla one in the
Langfuse UI, and a second copy of that logic here would be the same drift risk as
a second copy of the prompt.

### Project: Full `langchain_rag.py` with generation

```python
# Add to langchain_rag.py

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langfuse import observe
from langfuse.langchain import CallbackHandler

# _score_answer is private to ask.py, and imported anyway: the two reference-free
# scores are what make a LangChain answer comparable to a vanilla one in the
# Langfuse UI, and a second copy of that logic here would be the same drift risk
# as a second copy of the prompt.
from ask import LLM_MODEL, SYSTEM_PROMPT, TOP_K, _score_answer, assemble_prompt
from tracing import langfuse

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


# Update __main__ to add an "ask" mode
if __name__ == "__main__":
    import sys
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
```

Three shape decisions in there are worth understanding rather than copying.

**The chain returns `{answer, chunks, ...}`, not a string.** The obvious version
of `answer_question_lc_with_context` retrieves once for the chunks and then runs
a separate `PROMPT | model | parser` chain - which retrieves twice. Under the
pacing gate a second retrieval is a real 21-second wait, so on a 39-record eval
that mistake doubles the wall-clock. Two chained `RunnablePassthrough.assign`
calls keep the retrieved chunks in the dict flowing through the chain, so the
answer and the context it was built from come out together.

**`build_rag_chain` and `_chat_model` are `lru_cache`d.** Every `_store()` call
builds a fresh Chroma client and embeddings object, and `init_chat_model`
resolves and constructs a provider client. None of that is per-question work, and
paying it 39 times is pure latency.

**The system prompt is a `SystemMessage`, not a `("system", ...)` tuple.** A
tuple is parsed as an f-string-style template. Put a brace in `ask.py`'s prompt
one day and this file fails with a missing-variable error, in a module that had
nothing to do with the edit. A `SystemMessage` is passed through verbatim.

Test both a real question and an out-of-corpus one, to confirm citation and
refusal both survive the framework port:

```bash
uv run python langchain_rag.py ask "How often must internal audits be conducted?"
# Expect an answer citing a clause, e.g.
# (sqf-fundamentals-for-manufacturing-intermediate-09262019-ed-1-1-final.pdf, 2.5.5.1, p.28)

uv run python langchain_rag.py ask "What is the maximum fine for an OSHA violation?"
# Expect "Not found in the provided documents"
```

### Try streaming (one of LCEL's free wins)

```python
# Run this in a python REPL or a scratch script
from ask import assemble_prompt
from langchain_rag import PROMPT, _chat_model, _docs_to_chunks, get_retriever

question = "What must the food defense plan contain?"
chunks = _docs_to_chunks(get_retriever().invoke(question))

for part in (PROMPT | _chat_model()).stream(
        {"user_message": assemble_prompt(question, chunks)}):
    print(part.content, end="", flush=True)
```

You wrote zero streaming code, but `.stream()` works because every component in
the chain is a `Runnable`. That's the LCEL value proposition in one line.

Streaming the *generation* sub-chain rather than `build_rag_chain()` is
deliberate. The full chain's output is a dict (`{question, chunks, user_message, answer}`), so streaming it yields dict deltas rather than answer tokens - and its
first step is a retrieval that has to complete before any token can exist
anyway. Retrieve first, then stream the part that has something to stream.

Commit:

```bash
git add langchain_rag.py
git commit -m "Day 3 (wk5): full LCEL RAG chain, clause citations, streaming"
```

---

## Day 4 (Thu): Run the SAME Eval on the LangChain Version

This is the day that turns "I rebuilt it in LangChain" into "I measured both
implementations." That comparison is what makes this portfolio-grade.

### Concept: Reusing your eval harness

Your Week 3 eval harness calls `answer_question_with_context()` and expects back
`(answer, chunks)`, where each chunk carries `clause`, `page` and `source`. It
uses the chunks two ways: it computes `clause_hit@k` directly from them, and it
hands them to the judge so citation correctness and grounding can be scored
against the context the model actually saw. That is exactly why, on Day 3, the
LangChain answer function returns chunks and not just a string. Because it matches
that contract, the same golden set, the same judge, and the same deterministic
metrics run against the LangChain implementation unchanged. Same corpus, same
questions, same scoring. Pure apples to apples.

### Project: Eval the LangChain RAG

Make the import in `run_eval.py` an environment switch. Note it swaps
`answer_question_with_context`, the context-returning entry point, not the
text-only one:

```python
# In evals/run_eval.py, replace the `from ask import ...` line with this switch

import os
RAG_IMPL = os.getenv("RAG_IMPL", "raw")  # raw | langchain | langgraph

if RAG_IMPL == "raw":
    from ask import answer_question_with_context
elif RAG_IMPL == "langchain":
    from langchain_rag import answer_question_lc_with_context as answer_question_with_context
elif RAG_IMPL == "langgraph":
    from langgraph_rag import answer_question_graph_with_context as answer_question_with_context  # Day 5
else:
    raise ValueError(f"Unknown RAG_IMPL: {RAG_IMPL}")
```

The raw-API build also honors `RETRIEVAL_STRATEGY` (vanilla/hybrid/rerank). To
keep this comparison about the framework and not the retriever, leave
`RETRIEVAL_STRATEGY=vanilla` for both runs. The LangChain version above uses plain
vector retrieval, so vanilla is the matching baseline.

Run both:

```bash
RAG_IMPL=raw       RETRIEVAL_STRATEGY=vanilla uv run python evals/run_eval.py raw_api   "raw Anthropic SDK RAG, vanilla"
RAG_IMPL=langchain                            uv run python evals/run_eval.py langchain "LCEL RAG, same clause chunks, vanilla"
```

Then compare with your existing `compare()` helper:

```bash
uv run python evals/analyze.py evals/results/<raw_api>.csv evals/results/<langchain>.csv
```

### What to expect and what it means

Because you fed LangChain the identical clause chunks, the scores should be very
close: same corpus, same chunking, same model, same retriever. The residual
differences come from prompt-formatting details and from how each path assembles
context, not from anything fundamental. Watch three numbers specifically:

- `clause hit@3` should be nearly identical - retrieval is the same underneath.
- `citation` should be nearly identical - both use the same excerpt header and
  system prompt.
- `probe refusal` should hold - if it drops on the LangChain side, the prompt
  template dropped or reworded the refusal instruction, and that is a real
  finding worth writing down.

The headline for your writeup is usually: **the framework didn't meaningfully
change answer quality; it changed developer experience.** That's the honest,
interview-credible conclusion. Frameworks are about velocity and maintainability,
not magic quality gains. The one place a framework could have hurt you here is the
default chunker, and you sidestepped it deliberately.

Write 3-4 sentences of observations into `NOTES.md` for Day 7.

Commit:

```bash
git add evals/run_eval.py
git commit -m "Day 4 (wk5): RAG_IMPL switch, eval raw-API vs LangChain on identical golden set"
```

---

## Day 5 (Fri): Agentic RAG with LangGraph

### Concept: When you need LangGraph instead of LCEL

LCEL chains are directed and acyclic - data flows one way, start to finish. That's
perfect for "retrieve, then answer." But some workflows need:

- **Loops**: "retry until the output passes validation"
- **Branches**: "if the retrieved context is irrelevant, rewrite the query and search again"
- **State**: carrying information across multiple steps
- **Human-in-the-loop**: pausing for approval mid-flow

When you need any of those, you need LangGraph. LangGraph models your workflow as
a graph: nodes (functions that do work) and edges (transitions, which can be
conditional). LangChain components plug into LangGraph nodes, so nothing you
learned this week is wasted.

In 2026, most production agents are LangGraph applications that use LangChain
components internally. This is the on-ramp to Weeks 7-9.

### Concept: Agentic RAG, and why it fits a compliance corpus

Plain RAG retrieves once and answers. Agentic RAG can decide: is this retrieved
context good enough? If not, rewrite the query and retrieve again. That
decision-and-loop is the simplest meaningful agent, and it's a perfect first
LangGraph project.

It also has a specific payoff for this corpus. Your compliance system prompt
refuses when the retrieved excerpts don't contain the requirement. That is correct
for genuinely out-of-corpus questions, but it also fires when retrieval simply
missed a clause that is present. The grade-and-rewrite loop attacks exactly that
second case: it retries retrieval before the prompt gives up, which should reduce
false refusals on hard in-corpus questions while leaving true refusals on the
probes intact (a rewrite of an out-of-corpus question still finds nothing). You
will be able to see that in the eval: false-refusal rate down, probe refusal rate
unchanged.

The graph:

```
                  +-----------+
   query  ----->  | retrieve  |
                  +-----------+
                        |
                        v
                  +-----------+      not relevant
                  |  grade    | --------------------+
                  | context   |                     |
                  +-----------+                      v
                        | relevant            +--------------+
                        v                     | rewrite query|
                  +-----------+               +--------------+
                  | generate  |                     |
                  +-----------+                     | (loop back to retrieve)
                        |                           |
                        v                           |
                      answer  <----------- (after rewrite, retry)
```

### Project: `langgraph_rag.py`

Note what this file does **not** contain: no system prompt, no model string, no k,
no excerpt formatter. All four are imported, for the same reason Day 3 gave -
the Day 6 comparison is only about the loop, so everything else has to be
byte-identical to the other two implementations. The one new import is
`ask.format_excerpts`, split out of `assemble_prompt` for this file: the grader
needs the excerpts on their own, without the question and answer instruction
wrapped around them.

```python
"""Day 5: Agentic RAG with LangGraph. Retrieve -> grade -> (rewrite & retry | generate).

Carries the retrieved chunks in state so the answer function can return them for
the eval, matching ask.answer_question_with_context.

Everything that the Day 6 comparison holds constant is imported, not restated:
the system prompt, the model, k and the excerpt formatter all come from ask.py
via langchain_rag.py. The only thing this file adds is the loop - which is the
whole point of running it as a third implementation.
"""
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

import env  # noqa: F401  - .env before any os.environ read, same as every entry point
from ask import SYSTEM_PROMPT, TOP_K, _score_answer, assemble_prompt, format_excerpts
from langchain_rag import _chat_model, _docs_to_chunks, get_retriever
from tracing import langfuse

# Each retry costs a full retrieval, and under the Voyage pacing gate a
# retrieval is a real 21 seconds. Two attempts is already up to ~60s of waiting
# on a question the grader keeps rejecting; three would make a 39-record eval
# untenable on the free tier.
MAX_ATTEMPTS = 2


# The state carried through the graph
class RAGState(TypedDict):
    question: str
    original_question: str
    context: str
    chunks: list
    answer: str
    attempts: int
    relevant: bool


@lru_cache(maxsize=None)
def _grader():
    """Yes/no relevance judge. Same model as the answer path - a cheaper grader
    would be a reasonable optimization, but it would also mean the loop's
    decisions came from a different model than the comparison is about."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You judge whether the SQF documentation excerpts are "
                   "sufficient to answer the question. Reply with only 'yes' or 'no'."),
        ("human", "Excerpts:\n{context}\n\nQuestion: {question}\n\n"
                  "Are the excerpts sufficient?"),
    ])
    return prompt | _chat_model() | StrOutputParser()


@lru_cache(maxsize=None)
def _rewriter():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's question to be more specific and "
                   "retrieval-friendly for SQF certification documentation. Use the "
                   "vocabulary of the SQF code (clauses, requirements, records, "
                   "verification). Return only the rewritten question."),
        ("human", "{question}"),
    ])
    return prompt | _chat_model() | StrOutputParser()


@lru_cache(maxsize=None)
def _generator():
    """The compliance prompt, imported rather than paraphrased.

    SYSTEM_PROMPT arrives as a SystemMessage for the reason langchain_rag.py
    documents: a ("system", ...) tuple is parsed as a template, so a brace in
    ask.py's prompt would fail here. The human turn is built by
    ask.assemble_prompt, so the model sees the same bytes as the other two
    implementations - including the refusal instruction, which the eval's probe
    records depend on being present verbatim.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "{user_message}"),
    ])
    return prompt | _chat_model() | StrOutputParser()


def retrieve_node(state: RAGState) -> dict:
    docs = get_retriever(k=TOP_K).invoke(state["question"])
    chunks = _docs_to_chunks(docs)
    return {"chunks": chunks, "context": format_excerpts(chunks)}


def grade_node(state: RAGState) -> dict:
    """Ask the model whether the retrieved context can answer the question."""
    verdict = _grader().invoke(
        {"context": state["context"], "question": state["original_question"]}
    )
    return {"relevant": verdict.strip().lower().startswith("yes")}


def rewrite_node(state: RAGState) -> dict:
    """Rewrite the query to retrieve better context, then loop back.

    Rewrites the *current* question, not the original one. Rewriting the
    original every time makes attempt 2 re-issue attempt 1's query almost
    verbatim, so the second retrieval returns the same chunks the grader has
    already rejected and the retry is 21 seconds of nothing.
    """
    rewritten = _rewriter().invoke({"question": state["question"]}).strip()
    return {"question": rewritten, "attempts": state.get("attempts", 0) + 1}


def generate_node(state: RAGState) -> dict:
    """Generate with the same compliance prompt as ask.py: cite clauses, refuse when absent."""
    # Answer the user's ORIGINAL question, even though retrieval may have used a rewrite.
    user_message = assemble_prompt(state["original_question"], state["chunks"])
    return {"answer": _generator().invoke({"user_message": user_message})}


def should_continue(state: RAGState) -> str:
    """Conditional edge: generate if relevant or out of attempts, else rewrite."""
    if state.get("relevant") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "generate"
    return "rewrite"


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", should_continue, {"generate": "generate", "rewrite": "rewrite"})
graph.add_edge("rewrite", "retrieve")   # loop back after rewriting
graph.add_edge("generate", END)

app = graph.compile()


def _run(query: str) -> dict:
    """Invoke the graph with the Langfuse callback bound to the current trace.

    Constructed per call for the reason langchain_rag.py documents: the handler
    binds to whatever trace is open, so a module-level one would attach every
    question's spans to whichever trace happened to be first.
    """
    return app.invoke(
        {"question": query, "original_question": query, "attempts": 0},
        config={"callbacks": [CallbackHandler()]},
    )


@observe(name="rag-answer")
def answer_question_graph_with_context(query: str) -> tuple[str, list[dict]]:
    """Drop-in equivalent to ask.answer_question_with_context.

    Scored here rather than in the graph so the refusal and citation_grounding
    scores land on the same trace names the other two implementations use -
    without them the Day 6 false-refusal comparison has nothing to read.
    """
    result = _run(query)
    answer, chunks = result["answer"], result.get("chunks", [])
    # update_current_span, not update_current_trace - the latter is not on the
    # v3 client. Inside @observe this is the trace's root span anyway, so the
    # loop count is visible at the top of the trace where it is useful.
    langfuse.update_current_span(
        metadata={"rag_impl": "langgraph", "attempts": result.get("attempts", 0)},
    )
    _score_answer(answer, chunks)
    return answer, chunks


def answer_question_graph(query: str) -> str:
    return answer_question_graph_with_context(query)[0]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What has to happen when a critical limit at a CCP is exceeded?"
    result = _run(query)
    print(f"Q: {query}\n")
    print(f"Attempts: {result.get('attempts', 0)}")
    print(f"Retrieval question used: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    # Same reason as ask.py: the CLI exits before the SDK's background exporter
    # would have flushed, so an untraced run looks like a bug.
    langfuse.flush()
```

Run it on a hard, multi-clause question:

```bash
uv run python langgraph_rag.py "How do corrective action and verification requirements interact after a CCP deviation?"
```

Watch the output. On a hard multi-concept question you may see `Attempts: 1` - the
grader judged the first retrieval insufficient, rewrote the query, and retried.
That loop is the agent making a decision. Budget for the wait: each retry is a
second retrieval, and under the Voyage pacing gate a retrieval is a real 21
seconds.

Then run it on a probe to confirm the loop does not manufacture a false answer.
Use one of the five scored probes from `evals/golden.jsonl` rather than an
invented question, so what you see here is what the eval will score on Day 6:

```bash
uv run python langgraph_rag.py "What does ISO 22000 require for the qualifications of the food safety team leader?"
# probe-iso-22000-team-leader, tagged cross-standard
# Expect: Attempts: 2 - it burns both retries and still refuses with
# "Not found in the provided documents". Rewriting an out-of-corpus question
# just produces a more fluent out-of-corpus question.
```

### Visualize the graph (optional, nice for your README)

```python
# In a REPL
from langgraph_rag import app
print(app.get_graph().draw_mermaid())
```

It prints a Mermaid diagram of your graph that renders directly in a GitHub
README. The dotted edges out of `grade` are the conditional ones:

```
graph TD;
	__start__([__start__]):::first
	retrieve(retrieve)
	grade(grade)
	rewrite(rewrite)
	generate(generate)
	__end__([__end__]):::last
	__start__ --> retrieve;
	grade -.-> generate;
	grade -.-> rewrite;
	retrieve --> grade;
	rewrite --> retrieve;
	generate --> __end__;
```

Commit:

```bash
git add langgraph_rag.py ask.py langchain_rag.py
git commit -m "Day 5 (wk5): agentic RAG in LangGraph with grade-and-rewrite loop"
```

---

## Day 6 (Sat): Compare All Three Implementations

You now have three implementations of the same SQF RAG:

1. **Raw API** (`ask.py`) - everything by hand
2. **LCEL** (`langchain_rag.py`) - framework, linear chain
3. **LangGraph** (`langgraph_rag.py`) - framework, agentic with a loop

### Project: Three-way implementation comparison

The `RAG_IMPL` switch from Day 4 already handles all three. Run the LangGraph
version through the eval:

```bash
RAG_IMPL=langgraph uv run python evals/run_eval.py langgraph "agentic RAG, grade+rewrite loop"
```

Now you have three eval CSVs. Compare scores, but more importantly compare these
dimensions in a table for your writeup. For this corpus the rows that matter most
are the compliance metrics, so lead with them:

| Dimension             | Raw API          | LCEL   | LangGraph                          |
| --------------------- | ---------------- | ------ | ---------------------------------- |
| clause hit@3          | (yours)          | ~same  | same-or-better (retries retrieval) |
| Probe refusal rate    | (yours)          | ~same  | same (rewrite still finds nothing) |
| False refusal rate    | (yours)          | ~same  | lower (retries before giving up)   |
| Citation correctness  | (yours)          | ~same  | ~same                              |
| Overall (5-axis)      | (yours)          | ~same  | better on hard questions           |
| Lines of code         | ~205             | ~255   | ~190                               |
| Avg latency per query | (Langfuse)       | ...    | higher (extra grade/rewrite calls) |
| Avg cost per query    | ...              | ...    | higher (extra calls)               |
| Debuggability         | high (all yours) | medium | medium                             |
| Setup complexity      | low              | medium | medium-high                        |

Be honest about the lines-of-code row: on this corpus LCEL came out *longer* than
the raw API, not shorter. Two of this repo's constraints ate the savings. Voyage's
free tier forced `PacedVoyageEmbeddings` because `langchain_voyageai` builds its
own client with `max_retries=0`, and the single-source-of-truth rule meant the
LCEL path had to explain, in comments, every place it imports from `ask.py`
instead of restating. The generic "framework = fewer lines" claim assumes the
framework's defaults fit you. Here two of them did not, and working around a
default costs more than writing the thing yourself would have. That is a better
interview answer than the line count would have been.

The LangGraph version will cost more and run slower per query (it sometimes makes
2-3x the model calls) but should reduce false refusals and score better on the
hard multi-clause questions, because it can retry retrieval before the compliance
prompt refuses. That tradeoff - more cost and latency for fewer false refusals and
better hard-question handling - is exactly the kind of thing an SA discusses with
a customer. You'll have lived it.

### The key interview insight

Write this down and make it yours:

> Frameworks don't make answers better; they make development faster and unlock
> patterns (loops, branches, retries) that are painful to hand-roll. Choose raw
> API for simple, performance-critical, deeply-understood paths. Choose LCEL when
> you want velocity on standard patterns. Choose LangGraph when the workflow
> genuinely needs state, loops, or branching. On this compliance corpus the
> agentic version earned its higher cost specifically by lowering the
> false-refusal rate - it retried retrieval instead of letting the model give up -
> while leaving true refusals on out-of-corpus questions intact. On easy
> questions it just spent more for the same answer.

Commit:

```bash
git add evals/run_eval.py
git commit -m "Day 6 (wk5): three-way implementation comparison (raw / LCEL / LangGraph)"
```

---

## Day 7 (Sun): Document + Branch Strategy + Wrap

### Project: Organize the repo and write the comparison

Your `cert-rag-cli` repo now has three implementations living side by side. Make
that legible.

Add a `FRAMEWORKS.md` to the repo:

```markdown
# Three Implementations, One RAG

This repo implements the same SQF compliance RAG three ways, evaluated against an
identical 34-question golden set plus 5 refusal probes, scored on five axes
(factual, complete, relevant, citation, grounding) with deterministic clause-hit
and refusal metrics.

## 1. Raw Anthropic SDK (`ask.py`, `retrievers/`)
Everything hand-written: clause-aware chunking, embedding, three retrieval
strategies, prompt assembly, generation. Maximum control and understanding,
maximum code.

## 2. LangChain LCEL (`langchain_rag.py`)
The same pipeline using LangChain's composable Runnable interface and the pipe
operator. Reuses the clause chunks from chunk.py rather than LangChain's
RecursiveCharacterTextSplitter, because clause boundaries carry the citation
metadata. Roughly half the code. Streaming, batching, async for free.

## 3. LangGraph agentic RAG (`langgraph_rag.py`)
A stateful graph that grades retrieved context and rewrites the query to retry
when retrieval is insufficient. Higher cost and latency; lower false-refusal rate
and better on hard multi-clause questions.

## Results

(paste your three-way eval table here, compliance metrics first)

## When to use which

(your conclusions from Day 6)

## Note on the chunker

LangChain's RecursiveCharacterTextSplitter is a strict upgrade over naive
character chunking for prose. For this corpus it is the wrong default: it would
cut across clause boundaries and drop the clause/page metadata that every citation
depends on. All three implementations therefore share the clause-aware chunks from
chunk.py. Knowing when to override a framework's default is the point.
```

Update the main `README.md` to point at `FRAMEWORKS.md` and to note that the repo
demonstrates raw-API, LCEL, and LangGraph implementations. That breadth on one
repo is a strong signal: it says you understand the tools AND the fundamentals
underneath them.

### Project: serve the RAG over HTTP

One more thing belongs in the repo before you call it done, and it takes twenty
minutes: an HTTP endpoint. Everything so far is a CLI, and a CLI is not something
another system can call. The Drupal admin UI needs an endpoint; so does anything
else you would demo. This is also the smallest possible piece of evidence that
you think past the notebook - reviewers notice.

The entire design constraint is: **add no retrieval or generation logic**.
`serve.py` is a wrapper around `ask.answer_question_with_context` and nothing
else, so the CLI and the service can never drift into answering differently. If
you find yourself reimplementing prompt assembly here, stop - that is the same
single-source-of-truth failure the LangChain port avoided on Day 2.

```python
"""HTTP service exposing the SQF RAG pipeline for the Drupal admin UI.

A thin wrapper around ask.answer_question_with_context - it adds no retrieval or
generation logic of its own, so the CLI and the service always answer the same
way. Run locally with:

    uv run uvicorn serve:app --reload --port 8000

    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "How often are internal audits required?"}' | jq

Auth: if the RAG_API_KEY environment variable is set, every /ask request must
send a matching `X-API-Key` header. If it is unset, auth is disabled (fine for
local dev, not for the live site - set it in production).
"""
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ask import answer_question_with_context

app = FastAPI(title="cert-rag-cli", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    clause: str | None = None
    clause_title: str | None = None
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _check_auth(x_api_key: str | None) -> None:
    expected = os.environ.get("RAG_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _sources(chunks: list[dict]) -> list[Source]:
    """Retrieved chunks as citable sources, in rank order, deduplicated.

    Deduplicated because a long clause is sub-split into several chunks that
    share a clause and page, and retrieval routinely returns more than one of
    them - a UI listing the same citation repeatedly looks like a bug. Text is
    deliberately not returned: the answer already quotes what it relies on, and
    the excerpt bodies are large enough to dominate the response.
    """
    seen, out = set(), []
    for c in chunks:
        key = (c.get("source"), c.get("clause"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(Source(source=c["source"], clause=c.get("clause"),
                          clause_title=c.get("clause_title"), page=c.get("page")))
    return out


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None)) -> AskResponse:
    _check_auth(x_api_key)
    answer, chunks = answer_question_with_context(req.question)
    return AskResponse(answer=answer, sources=_sources(chunks))
```

```bash
uv run uvicorn serve:app --reload --port 8000

curl -s localhost:8000/health | jq
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How often are internal audits required?"}' | jq
```

Three decisions in a very small file, each worth being able to explain:

- **`sources` is deduplicated.** `MAX_CHARS` sub-splits a long clause into
  several chunks that share a clause and page, and retrieval routinely returns
  more than one of them. Passed through raw, the UI shows the same citation three
  times and looks broken. This is the clause-aware chunker's one sharp edge at
  the presentation layer, and it is worth knowing where it shows up.
- **Chunk text is not returned.** The answer already quotes what it relies on,
  and at `TOP_K=14` the excerpt bodies would dominate the response payload. The
  client needs to know *which clauses* backed the answer, not to re-read them.
- **Auth is opt-in via `RAG_API_KEY`.** Unset means no auth, which is right for
  local dev and wrong for the live site. Making it an env var rather than a
  config flag means the production posture is a deploy-time decision, not a code
  change - but say out loud that unset-means-open is the tradeoff you accepted.

Push:

```bash
git add FRAMEWORKS.md README.md serve.py
git commit -m "Day 7 (wk5): FRAMEWORKS.md comparison, HTTP service, README updated"
git push
```

### Week 5 Wrap-up Checklist

- [ ] `langchain_hello.py` - first LCEL chain
- [ ] `langchain_rag.py` - full LCEL RAG over the clause chunks, builds index, answers with citations, streams
- [ ] `langgraph_rag.py` - agentic RAG with grade-and-rewrite loop, returns chunks
- [ ] All three implementations evaluated on the same golden set, including citation and refusal metrics
- [ ] `FRAMEWORKS.md` comparison written, compliance metrics first
- [ ] `serve.py` answers over HTTP with deduped clause citations, and the CLI and
  the service share one code path
- [ ] You can write a basic LCEL chain (`prompt | model | parser`) from memory
- [ ] You can explain when to use LCEL vs LangGraph in one sentence
- [ ] You can explain `RunnableParallel` + `RunnablePassthrough` in the RAG pattern
- [ ] You can articulate the "frameworks change DX, not answer quality" insight, and the one place the default chunker would have hurt you
- [ ] You can explain why the agentic loop lowers false refusals without weakening true refusals

---

# WEEK 6: Adaptation - Fine-tuning, LoRA, QLoRA

**Goal**: Understand fine-tuning well enough to recommend it (or not) in a customer conversation, and prove you can do it by publishing one small LoRA adapter trained on your Spices Inc brand voice. This is the smallest necessary concession to "data science" - you will not become a data scientist, you will become conversant.

A note up front: this week uses **Google Colab's free T4 GPU**, not your local machine, unless `nvidia-smi` showed you a capable NVIDIA GPU. Fine-tuning needs a GPU and the free Colab tier is the universal path. Everything else (data prep, evaluation) happens locally.

---

## Day 8 (Mon): The Decision Tree (When to Fine-tune)

Heavy concepts, no GPU work. This is the most interview-relevant day of the week.

### Concept: What fine-tuning actually does

Fine-tuning continues training a pre-trained model on your own examples so it internalizes a behavior, style, or format. The base model already knows language; fine-tuning teaches it *your* task or voice.

Crucial framing for interviews: **fine-tuning teaches behavior, not facts.** It is poor at injecting new knowledge (that's what RAG is for) and good at shaping how the model responds (tone, format, structure, consistent style).

### Concept: Full fine-tuning vs LoRA vs QLoRA

Three techniques, increasing accessibility:

- **Full fine-tuning**: update every weight. Best results, but a 7B model needs ~84GB of GPU memory across multiple high-end GPUs. Also "destructive" - can cause catastrophic forgetting of the model's general skills. Almost never the right recommendation for an applied team.
- **LoRA (Low-Rank Adaptation)**: freeze the base model, train small adapter matrices (typically less than 1% of total parameters) injected at each targeted layer. Drastically less memory and time. The adapter is a tiny file (~100MB) you load alongside the base model. This is the default modern choice.
- **QLoRA (Quantized LoRA)**: LoRA on top of a base model compressed to 4-bit. Cuts memory further - a 7-8B model fine-tunes on a single 16GB GPU, which means the free Colab T4. The quality gap versus full fine-tuning is typically 1-2%, a negligible tradeoff for a 10x hardware reduction.

You'll use QLoRA because it runs on free hardware. Know all three names and the memory story for interviews.

### Concept: The decision tree (memorize this)

This is the single most valuable thing from Week 6 for interviews. When a customer asks "should we fine-tune?", you walk this tree:

```
Does the model need CURRENT or PRIVATE facts it doesn't know?
   YES -> RAG. Fine-tuning is the wrong tool for knowledge.
   NO  -> continue

Does the model need a consistent STYLE, FORMAT, or TONE that
prompting can't reliably enforce?
   Try PROMPTING first. If a good system prompt + few-shot examples
   gets you there -> stop, you're done, no fine-tuning needed.
   If prompting is inconsistent or the prompt gets too long/expensive
   -> LoRA fine-tuning is justified.

Do you need both current facts AND consistent behavior?
   -> RAG + fine-tuning. They compose; they're not mutually exclusive.

Is the task safety-critical or must structured output be guaranteed
at scale where even rare prompt failures are unacceptable?
   -> Fine-tuning (possibly full) plus guardrails.
```

The default answer for most business problems is "RAG or better prompting, not fine-tuning." Fine-tuning is the right tool less often than people expect, and saying so credibly signals senior judgment.

### Reading (1.5 hours)

- Hugging Face PEFT overview (LoRA/QLoRA): https://huggingface.co/docs/peft/index
- The "when NOT to fine-tune" framing in any current QLoRA guide. The key line to internalize: if RAG or better prompting solves it, that's cheaper and faster than fine-tuning.
- Optional deeper dive: Maxime Labonne's SFT + Unsloth walkthrough on the Hugging Face blog (search "mlabonne sft llama unsloth") - the gold-standard conceptual + practical reference.

No code today. Write the decision tree on a card and tape it next to the RAG pipeline drawing from earlier weeks.

---

## Day 9 (Tue): Build the Brand-Voice Dataset

Data quality is the whole game in fine-tuning. 500 excellent examples beat 50,000 mediocre ones. For learning, even 100-200 high-quality examples will produce a visible effect.

### Concept: Instruction dataset format

Fine-tuning a chat model needs examples in a consistent instruction/response shape. The common schema:

```json
{"instruction": "Write a product description for a spice blend.", "input": "Garam Masala - North Indian, coriander/cumin/black pepper base, warming", "output": "A traditional North Indian blend, hand-mixed in small batches..."}
```

The model learns to map (instruction + input) -> output in your voice.

### Project: Generate your brand-voice dataset

You have a real asset here: your actual Spices Inc product catalog and brand guide. The goal is a dataset of (product facts -> brand-voice description) pairs so the fine-tuned model learns to write in your voice from bare facts.

Two ways to build 150 examples:

**Option A (preferred, authentic): use your real product descriptions.** Export your existing product descriptions from the Spices Inc Drupal site. For each, the `output` is the real description (your actual voice), and the `instruction`/`input` is the bare facts. This is the highest-quality data because it's genuinely your voice.

**Option B (faster, synthetic): bootstrap with Claude.** Use your brand guide to have Claude generate descriptions in your voice for products you don't yet have copy for. Lower quality (it's imitating your voice, not your voice itself) but fast. A reasonable hybrid: use Option A for the examples you have and Option B to round out to 150.

Create `finetune/build_dataset.py`:

```python
"""Day 9: Build a brand-voice instruction dataset for fine-tuning."""
import json
from pathlib import Path

OUT = Path("finetune/brand_voice.jsonl")
OUT.parent.mkdir(exist_ok=True)

# Option A: hand-curated from your real catalog.
# Fill this list from your actual Spices Inc product descriptions.
# Each entry: bare facts in, real brand-voice description out.
examples = [
    {
        "instruction": "Write a product description in the Spices Inc brand voice.",
        "input": "Product: Garam Masala. Origin: North Indian. Base: coriander, cumin, black pepper, cardamom, cinnamon, clove. Heat: warming, not hot. Use: dals, curries, bloom in ghee.",
        "output": "A traditional North Indian blend, hand-mixed in small batches. The base of coriander, cumin, and black pepper is rounded with cardamom, cinnamon, and clove - warming rather than hot. Use it in slow-cooked dals and curries, or bloom it in ghee at the end of cooking.",
    },
    # ... add 149 more. Pull from your real catalog where you can.
]

with OUT.open("w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(examples)} examples to {OUT}")
```

Spend the bulk of today populating this. Aim for 150 examples. Pull real descriptions from your Drupal product catalog - you can export them with a quick Drush query or a View. This is the most valuable hour you'll spend this week; the dataset is the thing that determines whether the fine-tune works.

### Build a held-out test set

Reserve 15 of your examples as a test set (don't train on them). You'll use these on Day 12 to check whether the fine-tune actually learned your voice or just memorized.

```bash
# Quick split: last 15 lines to test, rest to train
head -n -15 finetune/brand_voice.jsonl > finetune/train.jsonl
tail -n 15 finetune/brand_voice.jsonl > finetune/test.jsonl
wc -l finetune/train.jsonl finetune/test.jsonl
```

Commit (the dataset is yours and safe to commit, but scrub anything proprietary first):

```bash
git add finetune/build_dataset.py finetune/brand_voice.jsonl finetune/train.jsonl finetune/test.jsonl
git commit -m "Day 9 (wk6): brand-voice instruction dataset, train/test split"
```

---

## Day 10 (Wed): Set Up Colab + Hugging Face

Lighter day. Get the fine-tuning environment ready so Day 11 is pure training.

### Step 1: Hugging Face account and token

1. Create an account at https://huggingface.co
2. Go to Settings -> Access Tokens -> Create a "write" token
3. Save it; you'll paste it into Colab tomorrow and use it to push your adapter

### Step 2: Open the Unsloth Colab notebook

Unsloth is the efficient fine-tuning library: 2x faster, ~70% less VRAM than vanilla Hugging Face training, and it fits 7-8B QLoRA on the free Colab T4. Rather than writing the notebook from scratch, start from their official one and adapt it.

1. Go to https://unsloth.ai/docs/get-started/unsloth-notebooks
2. Find the Llama 3.1 8B (or Phi-3-mini, smaller and faster) instruct QLoRA notebook
3. Open it in Google Colab
4. In Colab: Runtime -> Change runtime type -> T4 GPU (free tier)

### Step 3: Upload your dataset to Colab

In the Colab file browser, upload `finetune/train.jsonl`. (Or push it to a Hugging Face dataset and load by URL - either works.)

### Concept: What the notebook will do

Read through the notebook before running anything. The flow:

1. **Load a 4-bit base model** with `FastLanguageModel.from_pretrained(load_in_4bit=True)` - this is the Q in QLoRA
2. **Attach LoRA adapters** with `FastLanguageModel.get_peft_model(r=16, ...)` - the trainable ~0.5% of parameters
3. **Format your data** into the model's chat template
4. **Train** with `SFTTrainer` (Supervised Fine-Tuning) for a small number of steps
5. **Test** generation before/after
6. **Save** the adapter and optionally push to Hugging Face

You don't need to understand every line. You need to understand the five stages and be able to describe them. Read the notebook with the goal of being able to explain it, not memorize it.

No commit today; the work happens in Colab tomorrow.

---

## Day 11 (Thu): Fine-tune

The training day. Most of the time is the GPU working while you watch.

### Step 1: Adapt the notebook to your data

In the Unsloth notebook, replace the example dataset with yours. The relevant cell loads a dataset; point it at your uploaded `train.jsonl`. You'll need a formatting function that turns your `{instruction, input, output}` rows into the model's chat template. The notebook has an example formatting function; adapt it:

```python
def format_example(row):
    instruction = row["instruction"]
    user_input = row.get("input", "")
    output = row["output"]
    user_msg = f"{instruction}\n\n{user_input}".strip()
    # Use the model's chat template (the notebook shows the exact call)
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
```

### Step 2: Set conservative training parameters

For a 150-example dataset, you want a small run to avoid overfitting:

```python
# In the SFTConfig / TrainingArguments
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_steps = 5
max_steps = 60          # Small dataset, keep it short. Watch loss; stop if it plateaus.
learning_rate = 2e-4
logging_steps = 1
```

LoRA config (the defaults in the notebook are good):

```python
r = 16                  # rank - higher = more capacity, more overfitting risk
lora_alpha = 16
lora_dropout = 0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### Step 3: Run training

Execute the training cell. On a T4 with 150 examples and 60 steps, this takes roughly 10-20 minutes. Watch the loss in the logs - it should trend down. If it spikes or plateaus immediately, your data formatting is probably off.

### Step 4: Eyeball before/after

The notebook has inference cells. Run a held-out test prompt through the model and compare to the base model's output on the same prompt. You're looking for the fine-tuned version to sound more like your brand voice - shorter, more concrete, no forbidden words ("premium," "artisanal"), hyphens not em dashes.

This is the moment of truth. If the fine-tuned output clearly sounds more "you" than the base output, the LoRA worked.

### Step 5: Save and push the adapter

```python
# Save locally in Colab
model.save_pretrained("spice_voice_lora")
tokenizer.save_pretrained("spice_voice_lora")

# Push to your Hugging Face account (uses your write token)
model.push_to_hub("YOUR-HF-USERNAME/spice-voice-lora", token="hf_...")
tokenizer.push_to_hub("YOUR-HF-USERNAME/spice-voice-lora", token="hf_...")
```

Your adapter is now public on Hugging Face. That URL is a resume link.

Download the `spice_voice_lora` folder from Colab to your local machine too, so you have a copy.

---

## Day 12 (Fri): Evaluate the Fine-tune (Honestly)

Did it actually work, or does it just feel like it worked? Measure it the way you measured RAG.

### Concept: Evaluating a style fine-tune

You can't use exact-match - there's no single right description. Use the same LLM-as-judge muscle from Week 3, but score on brand-voice adherence instead of factual correctness.

Two things this section originally got wrong, both corrected below after the SQF run:

**The rubric has to come from your corpus, not from a template.** The placeholder brand rules this walkthrough used to ship forbade "gourmet" (used 43 times in the real 454-description catalog), forbade em dashes (57 of them), preferred five phrases that appear zero times, and preferred "building" over "bold" for heat when the corpus uses bold 67 times and building 3. Judging a fine-tune against invented rules scores the human-written source material as off-brand. Measure the rules first.

**Two arms is not enough.** Base vs. fine-tuned flatters the fine-tune and answers nothing a customer cares about. Run three.

### Project: Brand-voice judge

Three arms, all generated in Colab, all judged locally:

| Arm         | What it is                                                    |
| ----------- | ------------------------------------------------------------- |
| `base`    | Stock model, bare prompt                                      |
| `fewshot` | Stock model, measured brand rules + 3 exemplars in the prompt |
| `ft`      | The QLoRA adapter                                             |

A fine-tune earns its keep only if it beats a good prompt. If `ft` ≈ `fewshot`, the honest finding is that prompting already got you there - the most common real-world result, and the more interesting thing to write about.

**Where each piece runs.** The models run in Colab, because that is where the GPU is. The judge runs locally from the repo root, because that is where your Anthropic key and your catalog are. The handoff between them is one JSON file.

#### Step 1 (Colab): derive the brand rules from the corpus

Before writing any rubric, measure. Locally:

```bash
uv run python -c "
import json, re
recs = [json.loads(l) for l in open('finetune/brand_voice.jsonl')]
outs = [r['output'] for r in recs]; blob = '\n'.join(outs); n = len(recs)
print('avg words:', sum(len(o.split()) for o in outs)//n)
for pat in ['Flavor Profile', 'How To Use|How to Use', 'also known as', 'is popular with']:
    print(pat, f'{sum(1 for o in outs if re.search(pat, o))/n:.0%}')
"
```

Whatever comes back is your rubric. For the SQF run: 96% carry a Flavor Profile section, 96% a How To Use section, 82% open with the product name, ~293 words, almost no exclamation marks.

#### Step 2 (Colab): generate all three arms

Load the pushed adapter once and toggle it off for the base arms, so every arm comes from identical weights:

```python
import json
D = "/content/drive/MyDrive/spice-voice"
test  = [json.loads(l) for l in open(f"{D}/test.jsonl")]
train = [json.loads(l) for l in open(f"{D}/train.jsonl")]
shots = sorted(train, key=lambda r: len(r["output"]))[:3]   # shortest 3, to fit context

FastLanguageModel.for_inference(model)

def gen(text, n=600):
    ins = tokenizer([text], return_tensors="pt").to("cuda")
    out = model.generate(**ins, max_new_tokens=n, use_cache=True)
    s = tokenizer.batch_decode(out)[0].split("### Response:")[-1]
    return s.replace("<|end_of_text|>", "").replace("<|begin_of_text|>", "").strip()

bare    = lambda r: alpaca_prompt.format(r["instruction"], r["input"], "")
fewshot = lambda r: "\n\n".join(
    [alpaca_prompt.format(s["instruction"], s["input"], s["output"]) for s in shots] + [bare(r)])

res = {"names": [r["input"].splitlines()[0].replace("Name: ", "") for r in test]}
res["base"], res["fewshot"], res["ft"] = [], [], []

with model.disable_adapter():          # arms 1 and 2 - base weights
    for r in test:
        res["base"].append(gen(bare(r)))
        res["fewshot"].append(gen(fewshot(r)))

for r in test:                          # arm 3 - adapter on
    res["ft"].append(gen(bare(r)))

json.dump(res, open(f"{D}/outputs.json", "w"), indent=2)
```

Roughly 15 minutes for 45 generations. Writing to Drive rather than local Colab storage means an expired session costs nothing.

#### Step 3 (local): download and judge

Download `outputs.json` from Drive into `finetune/`, then:

```bash
uv run python finetune/eval_voice.py
```

See `finetune/eval_voice.py` for the full judge. Its shape:

- **Judge scores** from Claude via a forced tool call - `voice_adherence` and `structure_adherence` on 1-5, plus verbatim `filler_phrases` it caught.
- **Deterministic checks** with no API cost - word count, section presence.
- **A fabrication check**, which is the part the original walkthrough had no notion of and which turned out to matter most.

#### Step 4 (local): count the fabrications

A style fine-tune learns the *shape* of a section, not the facts inside it. In the SQF run the adapter reproduced the "How To Use" section perfectly and filled it with recipes that do not exist, plus a cross-sell to two products the company does not sell. That is not a bug to fix by training longer; it is the structural limit of the technique.

So grep every output for multi-word proper nouns and check them against the real catalog. Two rules, because the stakes differ:

- A name asserted in a cross-sell ("we also have X") must match a SKU **exactly**. "Thai Seasoning" is a fabrication even though "Spicy Thai Seasoning" exists - a customer clicking it finds nothing.
- Any other proper noun is flagged if it appears in no real description **and** introduces a word no SKU uses.

Measured on the SQF corpus that catches 4 of 5 known fabrications at an 11% false-positive rate on real copy. It is triage for a human, not a score - read the column, don't just sum it.

### The honest finding is the valuable finding

Whatever the deltas, the conclusion that demonstrates senior judgment is some version of:

> Fine-tuning moved voice adherence from X to Y on a 439-example dataset. A well-crafted prompt with 3 few-shot examples reached Z. The fine-tune is worth it when [voice consistency at scale matters / the prompt would otherwise be very long]; otherwise prompting is cheaper and easier to iterate. Neither arm fixes fabricated product names - that needs retrieval over the real catalog, with the adapter supplying only the voice.

That last sentence is the one most candidates never get to, because they never check.

Commit:

```bash
git add finetune/eval_voice.py finetune/results/
git commit -m "Day 12 (wk6): brand-voice judge, three-arm comparison"
```

---

## Day 13 (Sat): Write FINE_TUNING_NOTES.md

Document the whole thing as a portfolio artifact.

### Project: `finetune/FINE_TUNING_NOTES.md`

```markdown
# Fine-tuning a Brand-Voice LoRA Adapter

## Goal
Train a small model to write Spices Inc product descriptions in our brand
voice from bare product facts, without needing the full brand guide in the
prompt every time.

## Approach
- Base model: Llama-3.1-8B-Instruct (4-bit, QLoRA)
- Method: LoRA, r=16, on a single free Colab T4 GPU via Unsloth
- Dataset: 150 (facts -> brand-voice description) examples, 135 train / 15 test
- Training: 60 steps, ~15 minutes

## Results
- Base model voice adherence (LLM-judged, 1-5): X.XX
- Fine-tuned voice adherence: Y.YY
- Forbidden-word usage dropped from A to B

## What I learned
1. Fine-tuning teaches behavior (voice), not facts. For facts I'd use RAG.
2. Data quality dominated. My real catalog descriptions worked far better
   than synthetic ones.
3. (Honest finding about fine-tune vs prompting tradeoff)

## When I'd recommend this to a client
- Worth it when: voice consistency at scale, long brand guides that bloat
  every prompt, high-volume content generation.
- Not worth it when: good prompting already gets you there, dataset is small,
  or requirements change often (retraining cost).

## Artifact
LoRA adapter published: huggingface.co/YOUR-USERNAME/spice-voice-lora
```

Fill in your real numbers. This file plus the public Hugging Face adapter is your Week 6 deliverable.

Commit:

```bash
git add finetune/FINE_TUNING_NOTES.md
git commit -m "Day 13 (wk6): fine-tuning notes and conclusions"
git push
```

---

## Day 14 (Sun): Polish + LinkedIn + Wrap

### Update your Hugging Face model card

Go to your `spice-voice-lora` repo on Hugging Face and write a real model card (the README): what it is, what it was trained on (no proprietary data leaked), how to load it, the eval results. A good model card is the difference between "this person clicked through a tutorial" and "this person understands what they shipped."

### Optional LinkedIn post

You don't need to post every week, and Week 6 is a fine one to skip if the Week 4 post is still fresh. But if you do post, the fine-tune is genuinely interesting content because most people talk about fine-tuning without having done it:

> Fine-tuned my first LoRA adapter this week - taught an 8B model my company's product-description voice on a free Colab GPU using Unsloth + QLoRA. The more useful finding was the comparison: how much of the gain could a good system prompt have gotten on its own? Wrote it up here [link]. The decision of when to fine-tune vs when to just prompt better is the actual skill.

### Week 6 Wrap-up Checklist

- [ ] You can recite the fine-tune decision tree from memory
- [ ] You can explain LoRA vs QLoRA vs full fine-tuning, including the memory story
- [ ] 150-example brand-voice dataset built (ideally from real catalog data)
- [ ] One LoRA adapter trained on Colab and published to Hugging Face
- [ ] Brand-voice eval comparing fine-tuned vs base, honestly scored
- [ ] `FINE_TUNING_NOTES.md` written with the fine-tune-vs-prompt conclusion
- [ ] Hugging Face model card written
- [ ] You can answer "when would you fine-tune vs use RAG?" with a real example

---

## A Note on Pacing for Weeks 5-6

These two weeks are conceptually lighter than Weeks 3-4 but introduce a lot of new tooling surface (LangChain, LangGraph, Unsloth, Colab, Hugging Face). The risk isn't difficulty; it's version churn. LangChain especially breaks between minor versions, and a tutorial from six months ago may not run. When something doesn't work, check the version first - `uv pip freeze | grep langchain` - and consult the current docs rather than an old blog post.

If the Colab fine-tune fights you on Day 11 (CUDA errors, OOM, dependency conflicts are all common), don't lose the week to it. The Unsloth notebooks are maintained and usually run clean; start from their exact notebook and change only the dataset, not the training code. If you're truly stuck, a smaller base model (Phi-3-mini instead of Llama-3.1-8B) sidesteps most memory issues.

The strategic point of these two weeks: you can now speak both languages. You understand RAG and agents from first principles (Weeks 1-4) AND you can wield the industry-standard frameworks (Week 5) AND you've actually fine-tuned a model and can reason about when that's the right call (Week 6). That combination - fundamentals plus frameworks plus judgment about when to use what - is exactly the profile that gets AI Solutions Engineer offers.

Weeks 7-9 are next, and they're the payoff: real agents, MCP, and the portfolio centerpiece. You're well-positioned now. Keep moving.
