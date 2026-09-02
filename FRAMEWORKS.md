# Three Implementations, One RAG

This repo implements the same SQF compliance RAG three ways, evaluated against an
identical 34-question golden set plus 5 refusal probes, scored on five axes
(factual, complete, relevant, citation, grounding) with deterministic clause-hit
and refusal metrics.

The comparison is deliberately narrow. Corpus, chunks, embedding model, system
prompt, answer model and `TOP_K=14` are identical in all three arms, and the
prompt and excerpt formatter are *imported* from `ask.py` rather than
paraphrased, so the model sees the same bytes every time. The only variable is
the framework the pipeline is written in.

Full results and caveats: Experiment 4 of [`evals/EVAL_REPORT.md`](evals/EVAL_REPORT.md).

## 1. Raw Anthropic SDK (`ask.py`, `retrievers/`)

Everything hand-written: clause-aware chunking, embedding, three retrieval
strategies, prompt assembly, generation. Maximum control and understanding,
maximum code. One retrieval and one model call per question.

## 2. LangChain LCEL (`langchain_rag.py`)

The same pipeline using LangChain's composable Runnable interface and the pipe
operator. Reuses the clause chunks from `chunk.py` rather than LangChain's
`RecursiveCharacterTextSplitter`, because clause boundaries carry the citation
metadata. Streaming, batching and async come for free with the Runnable
interface, though this pipeline uses none of them.

## 3. LangGraph agentic RAG (`langgraph_rag.py`)

A stateful graph that grades retrieved context and rewrites the query to retry
when retrieval is insufficient, capped at two rewrites. The back-edge from
`rewrite` to `retrieve` is the thing an LCEL chain cannot express: a chain is a
pipe and cannot feed its own input. A question that exhausts its retries costs
three retrievals and six model calls against the raw path's one and one.

## Results

Three full 39-record runs, `TOP_K=14`, vanilla retrieval in every arm.

| Metric           | Raw API | LCEL | LangGraph |  L-R |  G-R |
| ---------------- | ------- | ---- | --------- | ---- | ---- |
| probe refusal %  |   100.0 | 100.0|     100.0 | +0.0 | +0.0 |
| false refusal %  |     2.9 |  2.9 |       2.9 | +0.0 | +0.0 |
| cites expected % |    97.1 | 97.1 |      97.1 | +0.0 | +0.0 |
| hit@1 %          |    85.3 | 85.3 |      85.3 | +0.0 | +0.0 |
| hit@3 %          |    97.1 | 97.1 |      97.1 | +0.0 | +0.0 |
| hit@5 %          |    97.1 | 97.1 |      97.1 | +0.0 | +0.0 |
| citation         |    4.56 | 4.53 |      4.65 |-0.03 |+0.09 |
| grounding        |    4.82 | 4.82 |      4.94 |+0.00 |+0.12 |
| factual          |    4.76 | 4.65 |      4.76 |-0.12 |+0.00 |
| complete         |    4.29 | 4.29 |      4.24 |+0.00 |-0.06 |
| relevant         |    4.47 | 4.38 |      4.41 |-0.09 |-0.06 |
| **overall**      |    4.58 | 4.54 |      4.60 |-0.05 |+0.02 |

By difficulty: easy 4.78/4.73/4.82, medium 4.40/4.48/4.57, hard 4.58/4.40/4.42.

| Dimension              | Raw API | LCEL   | LangGraph    |
| ---------------------- | ------- | ------ | ------------ |
| Lines in the module    |     205 |    255 |          191 |
| Mean latency / query   |   17.2s |  17.5s |        23.9s |
| ...scored records only |   16.8s |  17.2s |        18.3s |
| ...probe records only  |   20.2s |  19.5s |        62.3s |
| Full-run wall clock    | 11.2min |11.3min |      15.6min |
| Input tokens / query   |      1x |    ~1x | ~2.15x (est) |
| Debuggability          |    high | medium |       medium |
| Setup complexity       |     low | medium |  medium-high |

The compliance metrics are the top block on purpose: in a food-safety context,
refusing correctly and citing the governing clause matter more than a judge's
mean. On those metrics the three arms are indistinguishable - identical to the
decimal on all six.

Four things the numbers say that the going-in expectations did not:

1. **The agentic loop did not change retrieval at all.** It fired on 6 of 39
   records - all 5 probes plus `corrective-action-m1`. On the other 33 the grader
   judged the first context sufficient, so retrieval was a single vanilla call,
   exactly as in the other two arms. A retry loop only helps where the grader
   rejects something, and at this corpus and k it almost never does.
2. **False refusal did not improve.** All three arms sit at 2.9% - the same
   single false refusal, on a record where the grader accepted the first context
   and the loop never ran.
3. **LangGraph is worse on hard questions, not better.** Hard drops 4.58 -> 4.42
   while medium gains 4.40 -> 4.57. Both movements clear the 0.06 judge-noise
   floor; the overall mean hides them.
4. **The loop's cost lands where it cannot help.** Probes cost LangGraph 42s more
   per record than raw; scored records cost 1.5s more. An out-of-corpus question
   is exactly the case where the grader correctly says "insufficient" and the
   rewrite correctly finds nothing, so every probe pays a full retry to reach the
   refusal the other arms reach immediately.

The one clear win is `corrective-action-m1`, 2.60 -> 3.60 - the single scored
question that looped. The mechanism works. This run says little about how often
it gets the chance.

## When to use which

**Raw SDK when the pipeline is the product.** It stays in production here. It is
within noise of both alternatives on every compliance metric, retrieves
identically, costs less than half the input tokens of the agentic arm, and when
something breaks the stack trace lands in code this repo owns. For a pipeline
this shape - retrieve once, assemble a prompt, generate - a framework abstracts
over about fifty lines of orchestration and adds a dependency whose failure modes
you then have to learn. `PacedVoyageEmbeddings` in `langchain_rag.py` is that
tax made concrete: `langchain_voyageai.VoyageAIEmbeddings` builds its own client
with `max_retries=0` and forbids extra constructor kwargs, so the first 429
propagates out of `retriever.invoke()` and the whole class had to be rewritten to
route through this repo's rate-limit gate.

**LCEL when you want what the Runnable interface gives you.** Streaming, batching,
async, and a swappable model layer are real, and they are free once the pipeline
is expressed as a chain. This pipeline uses none of them, which is why LCEL shows
up here as pure overhead: -0.05 overall, no metric moved, more code than the raw
implementation it replaces. That is a fact about this pipeline, not about LCEL.

**LangGraph when the control flow is genuinely a graph.** Cycles, conditional
branching and durable state are things a chain cannot express, and writing them
by hand is where hand-rolled orchestration starts costing more than it saves. The
loop earns its keep when retrieval failure is common enough that retrying pays
for itself. Here it is not: one scored question in 34 improved, and the ~2.15x
input tokens and 15.6-minute run bought +0.02 overall. Before reaching for it,
measure how often your grader would actually reject the first retrieval - if that
number is near zero, the loop is a cost with no counterparty.

Two changes would make the agentic arm defensible on this corpus: grade against
a cheaper model, or on the question alone rather than the full 14-excerpt context
(the grading call is why input tokens roughly double on *every* query, looped or
not); and skip the loop on questions the corpus plainly does not cover, where the
retry only slows a refusal that was already correct.

## Note on the chunker

LangChain's `RecursiveCharacterTextSplitter` is a strict upgrade over naive
character chunking for prose. For this corpus it is the wrong default: it would
cut across clause boundaries and drop the clause/page metadata that every citation
depends on. All three implementations therefore share the clause-aware chunks from
`chunk.py`. Knowing when to override a framework's default is the point.

## Note on the line counts

The modules are not independent, so the counts above are increments, not
standalone sizes. `langchain_rag.py` imports the prompt, model, `TOP_K`, excerpt
formatter and scoring from `ask.py`, and the rate-limit gate from
`retrievers/embed.py`. `langgraph_rag.py` imports from both - its retriever, chat
model and document conversion all come from `langchain_rag.py`. So LangGraph's
191 lines are what the loop costs *on top of* the other two, not a self-contained
implementation, and the raw arm's 205 excludes the 167 lines of
`retrievers/embed.py` and `retrievers/vanilla.py` it depends on. Read the row as
"lines you would write to add this arm", and note that the arm ordered last
benefits most.

## Note on what these numbers do not settle

The runs are a week apart (raw and LCEL on 2026-08-24, LangGraph on 2026-08-31),
so ordinary between-day judge variance applies on top of the 0.06 noise floor.
Latency is wall clock under a 21s-per-call rate limiter, so it measures the Voyage
free tier at least as much as it measures the pipelines - the scored-only row is
the usable comparison. Cost is estimated from the call graph, not measured: the
result CSVs carry no token counts and the local Langfuse build's read API is
unavailable. And every per-difficulty figure rests on 11 or 12 questions.
