# Weeks 1-2 Deep Dive (SQF corpus): LLM Fundamentals and RAG From Scratch

**For the `cert-rag-cli` build. New corpus: your company's SQF certification
documents (PDFs and Word files). Working tutorial, not a plan. Aim: by end of
Week 2 you have made 50+ raw API calls and shipped one public RAG system you
understand line by line.**

Self-contained. Week 1 builds LLM fundamentals in a scratchpad repo; Week 2
builds the SQF RAG in `cert-rag-cli`.

---

## Before You Start: Pre-flight Setup (90 minutes, one-time)

### 1. Confirm your WSL2 environment is the workspace

All Python work happens inside WSL2, not Windows-native Python. Better
performance, better library compatibility, and your Claude Code workflow already
lives there. Open a WSL2 terminal:

```bash
# Verify you're in WSL2, not PowerShell
uname -a
# Should say "Linux ... microsoft-standard-WSL2 ..."

# Verify you're in your WSL2 home (not /mnt/c/...)
pwd
# Should be /home/yourname, not /mnt/c/Users/...
```

Do not put AI projects under `/mnt/c/...`. The WSL2-to-Windows filesystem bridge
is slow and causes hangs during embedding work. Everything under `~/projects/`.

```bash
mkdir -p ~/projects
cd ~/projects
```

### 2. Install `uv`

`uv` replaces pip, pip-tools, pyenv, and virtualenv with one fast binary. Coming
from PHP/Composer, think of it as Composer for Python but faster. Skip this if
you already have it from the prior run.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### 3. Get API keys and store them safely

Two accounts for Week 1, $20 each, both with free tiers. Voyage gets added in
Week 2.

Anthropic: console.anthropic.com -> your name (top right) -> API Keys -> create
one, name it `cert-rag-2026`, copy it immediately (you cannot view it again), add
$20 credit under Plans & Billing.

OpenAI: platform.openai.com -> Settings -> API keys -> create a secret key, same
drill, add $20 credit. OpenAI is used only for the Day 1 parity exercise and is
optional if you would rather stay single-vendor.

Store the keys in a gitignored `.env` at the root of each project, loaded with
`python-dotenv`. This is the pattern the whole plan uses: from Week 2 the
`cert-rag-cli` project formalizes it in a tiny `env.py` module (built in Week 3),
and a fresh terminal, a new day, or a clone all start from the same credentials
instead of whatever was last exported by hand. No `source` step, and nothing
lives in your shell profile.

For this Week 1 scratchpad, create `.env`:

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=          # fill in Week 2
EOF
grep -qxF '.env' .gitignore 2>/dev/null || echo '.env' >> .gitignore
```

Load it at the top of any script that reads a key:

```python
from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ; real env vars still win
```

`uv add python-dotenv` is included in the dependencies below. Real environment
variables always take precedence, so CI or production that sets them properly is
unaffected by a stray `.env` on disk.

### 4. Create the Week 1 scratchpad project

Week 1 is corpus-independent LLM practice. It lives in its own throwaway repo so
your `cert-rag-cli` portfolio repo stays clean. If you still have
`llm-fundamentals` from the prior run, either reuse it or start fresh; the steps
below assume fresh.

```bash
cd ~/projects
mkdir llm-fundamentals
cd llm-fundamentals

uv init --python 3.12
uv add anthropic openai python-dotenv

uv run python -c "import anthropic, openai; print('ok')"   # prints: ok
```

From here you run Python with `uv run python script.py`. `uv` manages the
virtual environment for you.

### 5. Initialize git and a GitHub repo for the scratchpad

```bash
git init
printf '%s\n' '__pycache__/' '.env' '*.pyc' > .gitignore
git add .
git commit -m "Initial commit: uv-managed Python project for LLM fundamentals"
```

Create a public repo `llm-fundamentals` on GitHub, then:

```bash
git remote add origin git@github.com:YOUR-HANDLE/llm-fundamentals.git
git branch -M main
git push -u origin main
```

Low stakes, no README required. Your `cert-rag-cli` project is separate and,
per your setup, already scaffolded with uv and git; you configure it on Week 2
Day 9.

---

## WEEK 1: LLM API Fundamentals

Goal: stop being someone who uses AI tools and become someone who builds with
them. Eight short scripts, each one concept. You should understand every line.

---

### Day 1 (Mon): Concepts + First Call

**Concept 1: The Messages API**

Every modern LLM API works the same way: you send a list of messages, you get
back a response. A message has a `role` (who said it) and `content` (what they
said). Roles are `user`, `assistant`, and `system`.

```
system:    "You are a helpful food-safety expert."
user:      "What is a HACCP plan?"
assistant: "A HACCP plan is a documented system for controlling food safety hazards..."
```

When you make an API call, you send the entire conversation so far. The model
has no memory between calls. If you want continuity, you pass the history back
every time. This is the single most important mental model. The "stateless API +
client-managed history" pattern is how every conversational AI works under the
hood.

**Concept 2: Tokens**

Tokens are the units of LLM pricing and context. A token is roughly 4 characters
of English, or about 0.75 words. "spices" is 1 token. "internationalization" is
4 tokens. The number `1234567` is 2 tokens; `1,234,567` is 5. Two reasons it
matters: pricing (you pay per input token and per output token, output typically
4-5x the input price) and the context window (each model has a max size, 200K for
Claude Sonnet 4.6; exceed it and the call is rejected). You do not count tokens
by hand; both SDKs report usage in every response.

**Concept 3: Temperature**

A number from 0 to 1 (Anthropic) or 0 to 2 (OpenAI) controlling randomness.
Temperature 0 is deterministic-ish (same input, same output most of the time).
Temperature 1 is creative. Use 0 for extraction, classification, code
generation. Use 0.7-1.0 for brainstorming and prose. Most production systems run
at 0.0 or very low.

**Project 1.1: `hello.py` (first call)**

Create `hello.py` in `~/projects/llm-fundamentals/`:

```python
"""Day 1: First API call. The minimum viable LLM script."""
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # read .env into os.environ (from Week 2, cert-rag-cli does this in env.py)

client = Anthropic()  # Reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "In two sentences, what is SQF certification?"}
    ]
)

print(response.content[0].text)
print(f"\nTokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
```

Run it:

```bash
uv run python hello.py
```

You should see a two-sentence answer and a usage line like `Tokens used:
input=16, output=54`. If you get an authentication error, your env var is not
loaded; check your `.env` has the key and that the script calls `load_dotenv()`.

**Project 1.2: Add OpenAI parity**

Make `hello_openai.py`:

```python
"""Same call, different vendor. Notice the API parity."""
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "In two sentences, what is SQF certification?"}
    ]
)

print(response.choices[0].message.content)
print(f"\nTokens used: input={response.usage.prompt_tokens}, output={response.usage.completion_tokens}")
```

Run it and compare outputs.

Key takeaway: the two APIs are 90% the same shape with minor naming differences
(`messages.create` vs `chat.completions.create`, `content[0].text` vs
`choices[0].message.content`, `input_tokens` vs `prompt_tokens`). By end of Week
1 you should be able to write either from memory.

```bash
git add hello.py hello_openai.py
git commit -m "Day 1: first API calls, Anthropic and OpenAI parity"
```

---

### Day 2 (Tue): System Prompts and Multi-turn Conversations

**Concept: System prompts**

A system prompt tells the model who it is, how to behave, and what constraints
apply. It is the most powerful prompt-engineering lever you have. Move work into
the system prompt whenever you can, because it is the most stable across turns.

**Concept: Multi-turn**

To have a conversation, you build up the messages list across turns. The model
sees the entire history every time.

**Project 2.1: `chat_loop.py`**

```python
"""Day 2: A working chat loop with system prompt and history."""
from anthropic import Anthropic

client = Anthropic()

system_prompt = """You are a senior engineer mentoring someone moving into applied AI work,
with a background in food manufacturing and quality systems.
Be direct and concise. Assume technical fluency. Use plain hyphens, never em dashes."""

conversation = []  # This is the history we'll grow

print("Chat with the mentor. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break

    conversation.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation
    )

    assistant_text = response.content[0].text
    print(f"\nMentor: {assistant_text}\n")

    # Critical: append the assistant turn so the next user message has context
    conversation.append({"role": "assistant", "content": assistant_text})
```

Run it and have a conversation. Ask three questions where the third depends on
the second (e.g., "What is a HACCP plan?" -> "Give me a simple example." -> "Now
turn that into a checklist."). Watch how it carries context.

Now break it on purpose. Delete the `conversation.append({"role": "assistant",
...})` line and re-run. The third question is useless because the assistant never
remembers what it said. The lesson: the model's memory is whatever you pass in.
Nothing more.

```bash
git add chat_loop.py
git commit -m "Day 2: multi-turn chat with system prompt and history management"
```

---

### Day 3 (Wed): Temperature Sweep

**Project 3.1: `temperature_sweep.py`**

Goal: see for yourself how temperature changes output. Interviewers ask "when
would you use temperature 0 vs 1?" and the answer lands better if you have run
the experiment.

```python
"""Day 3: Run the same prompt at multiple temperatures and compare."""
from anthropic import Anthropic

client = Anthropic()

prompt = "Write a one-sentence tagline for a small-batch spice company."

for temp in [0.0, 0.3, 0.7, 1.0]:
    print(f"\n=== Temperature {temp} ===")
    for run in range(3):  # Run each temp 3 times
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"Run {run+1}: {response.content[0].text.strip()}")
```

That is 12 calls. Read the output carefully. Two things you should observe:
temperature 0 produces nearly identical outputs across runs (some sampling
variation, but close), and temperature 1.0 produces wildly different outputs.

Interview-ready takeaway: temperature 0 for anything where consistency matters
(extraction, classification, structured output, code). Higher temperatures for
creative work where variety is the point.

```bash
git add temperature_sweep.py
git commit -m "Day 3: temperature sweep experiment"
```

---

### Day 4 (Thu): Tool Use (The Core Pattern)

The most important day of Week 1. Tool use is the foundation of agents, and
agents are the foundation of most AI Solutions Engineer work.

**Concept: What tool use actually is**

Despite the name, the model never uses a tool. It emits a structured request that
says "call this function with these arguments." Your code calls the function,
sends the result back, and the model continues using that result.

```
1. You define a tool with a JSON schema (name, description, parameters).
2. You include the tool definition in your API call.
3. The model decides whether to call it. If yes, it returns a tool_use block.
4. You execute the tool yourself (it's your code; you control what happens).
5. You send the result back in a tool_result block.
6. The model produces its final response using the tool result.
```

A turn-based dance, not magic. The model decides when to call which tool with
what arguments. You execute and report back.

**Project 4.1: `weather_tool.py`**

Fake weather function so you need no external API key. The pattern is identical
to a real one.

```python
"""Day 4: Tool use loop. The model decides; your code executes."""
import json
from anthropic import Anthropic

client = Anthropic()

# 1. Define the tool with a JSON schema
weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather for a US city. Returns temperature in Fahrenheit and conditions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Kingston, PA'"},
        },
        "required": ["city"]
    }
}

# 2. Implement the actual function (fake data for now)
def get_weather(city: str) -> dict:
    """Pretend to call a weather API."""
    fake_data = {
        "Kingston, PA": {"temp_f": 62, "conditions": "partly cloudy"},
        "San Francisco, CA": {"temp_f": 58, "conditions": "foggy"},
        "Phoenix, AZ": {"temp_f": 95, "conditions": "sunny and dry"},
    }
    return fake_data.get(city, {"temp_f": 70, "conditions": "unknown"})

# 3. The conversation loop
messages = [
    {"role": "user", "content": "Should I wear a jacket in Kingston PA today?"}
]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[weather_tool],
        messages=messages
    )

    print(f"\n[stop_reason: {response.stop_reason}]")

    # If the model is done, print and exit
    if response.stop_reason == "end_turn":
        print(f"\nFinal answer: {response.content[0].text}")
        break

    # If the model wants to use a tool, execute it
    if response.stop_reason == "tool_use":
        # Append the assistant's turn (including the tool_use block)
        messages.append({"role": "assistant", "content": response.content})

        # Find the tool_use block and execute
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                print(f"-> Model called {tool_name}({tool_input})")

                if tool_name == "get_weather":
                    result = get_weather(tool_input["city"])
                    print(f"-> Result: {result}")

                    # Send the result back
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        }]
                    })
```

Run it. You should see:

```
[stop_reason: tool_use]
-> Model called get_weather({'city': 'Kingston, PA'})
-> Result: {'temp_f': 62, 'conditions': 'partly cloudy'}

[stop_reason: end_turn]
Final answer: At 62F and partly cloudy in Kingston, a light jacket would be comfortable...
```

That dance between the model and your code is what every agent is doing under the
hood. Variations get you to multi-tool agents, retrieval-augmented agents, and
the whole MCP ecosystem.

Expand the project: add a second tool, `get_clothing_recommendation(temp_f,
conditions)`, that returns a sentence. Re-run with "Should I wear a jacket?" and
the model should chain both calls.

```bash
git add weather_tool.py
git commit -m "Day 4: tool use loop with single tool, then chained tools"
```

---

### Day 5 (Fri): Structured Outputs

**Concept**

A common production need: get the model to return data in a strict shape your
code can parse. The naive approach is "tell the model to return JSON" and hope.
The robust approach is to define a schema and have the SDK enforce it. Anthropic
does this via tool use (define a tool whose `input_schema` is the shape you want
and force the model to call it). OpenAI has a first-class `response_format`
parameter. Both work; the patterns are similar.

**Project 5.1: `extract_orders.py`**

Processing customer service emails and extracting structured data. Exactly the
kind of thing an SA hiring manager asks you to demo.

```python
"""Day 5: Structured output via the Anthropic tool-use trick."""
import json
from anthropic import Anthropic

client = Anthropic()

# The shape we want the model to produce
extract_tool = {
    "name": "extract_order_details",
    "description": "Extract structured order details from a customer email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_name": {"type": "string"},
            "order_number": {"type": "string", "description": "If not present, use empty string"},
            "issue_type": {
                "type": "string",
                "enum": ["missing_item", "wrong_item", "damaged", "late_delivery", "refund_request", "other"]
            },
            "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
            "summary": {"type": "string", "description": "One-sentence summary of the issue"}
        },
        "required": ["customer_name", "issue_type", "urgency", "summary"]
    }
}

email_text = """
Hi - my name is Maria Vasquez. I ordered the garam masala blend (order #SP-44219) last week and
when it arrived yesterday the jar was cracked and half the contents had leaked out. I have a
dinner party Saturday and really need a replacement. Can you ship one overnight?
"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=[extract_tool],
    tool_choice={"type": "tool", "name": "extract_order_details"},  # Force the tool
    messages=[
        {"role": "user", "content": f"Extract order details from this email:\n\n{email_text}"}
    ]
)

# The structured data is in the tool_use block
for block in response.content:
    if block.type == "tool_use":
        data = block.input
        print(json.dumps(data, indent=2))
```

Run it. You should get clean structured output:

```json
{
  "customer_name": "Maria Vasquez",
  "order_number": "SP-44219",
  "issue_type": "damaged",
  "urgency": "high",
  "summary": "Customer received damaged jar of garam masala and needs overnight replacement for Saturday event."
}
```

The `tool_choice={"type": "tool", "name": ...}` parameter forces the model to
call this specific tool every time, which guarantees a structured response in
your defined shape.

Why this matters: this is the primitive behind every "extract X from
unstructured text" workflow, every "categorize this support ticket," every "fill
out this form from a conversation." You will use it constantly.

Optional variant that previews Week 2: swap the schema to
`{clause_number, requirement, frequency}` and paste in a paragraph from one of
your SQF clauses instead of the email. Same forced-tool-use mechanics, and it
warms up the clause-parsing intuition you use on Day 10.

```bash
git add extract_orders.py
git commit -m "Day 5: structured output via forced tool use"
```

---

### Day 6 (Sat): Prompt Caching (Cost Control)

**Concept**

When you make calls with a long, stable prefix (a big system prompt, a long
document, a knowledge base), you can mark that prefix cacheable and Anthropic
stores the processed form for about 5 minutes. Subsequent calls that hit the
cache pay roughly 10% of the input price for that portion. This is the single
biggest cost lever in production LLM apps, and it separates people who have
shipped from people who have only prototyped.

**Project 6.1: `caching_demo.py`**

```python
"""Day 6: Prompt caching for cost savings on repeated context."""
from anthropic import Anthropic

client = Anthropic()

# Imagine this is a big document we'll reference repeatedly
long_context = """
[Spices Inc Brand Guide]
Voice: warm, expert, never patronizing. Use plain language. Avoid superlatives.
Never use em dashes - use regular hyphens instead.
Audience: home cooks who care about quality but are not professionals.
Forbidden words: "elevate", "premium", "artisanal", "gourmet".
Preferred words: "well-made", "carefully sourced", "small batch", "honest".

[Product Catalog Highlights]
- Garam masala: hand-mixed, North Indian style, coriander/cumin/black pepper base.
- Berbere: Ethiopian blend, chili-forward, includes fenugreek and cardamom.
- Za'atar: Levantine, sumac/sesame/thyme, finishing spice.
""" * 5  # Make it longer to clearly exceed the caching minimum

questions = [
    "Write a one-line product tagline for our garam masala.",
    "Suggest a recipe blurb featuring our za'atar.",
    "What's a good email subject line for a berbere promotion?"
]

for i, q in enumerate(questions, 1):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": long_context,
                "cache_control": {"type": "ephemeral"}  # Mark this as cacheable
            }
        ],
        messages=[{"role": "user", "content": q}]
    )

    u = response.usage
    print(f"\n--- Q{i} ---")
    print(f"Q: {q}")
    print(f"A: {response.content[0].text.strip()}")
    print(f"Tokens: input={u.input_tokens}, output={u.output_tokens}, "
          f"cache_read={getattr(u, 'cache_read_input_tokens', 0)}, "
          f"cache_create={getattr(u, 'cache_creation_input_tokens', 0)}")
```

Run it. Watch the token usage line. The first call shows `cache_create` populated
(you paid to write to cache). The second and third show `cache_read` populated
(you read from cache, paying about 10% of the normal rate for those tokens).

What you demonstrated: a real cost optimization. A customer-service bot with a
5K-token system prompt asking 100 questions an hour can pay far less with caching
than without, same answers, same model, different bill.

```bash
git add caching_demo.py
git commit -m "Day 6: prompt caching demo with cache_read/cache_create usage"
```

---

### Day 7 (Sun): Cost Calculator + Reading Day

Light coding day. Build one small utility and read the docs you have been putting
off.

**Project 7.1: `cost_calc.py`**

```python
"""Day 7: A pocket calculator for estimating LLM costs."""

# Prices in dollars per million tokens. These change - CHECK CURRENT PRICING at
# https://www.anthropic.com/pricing before quoting any of these numbers.
PRICES = {
    "claude-opus-4-7":    {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    "claude-sonnet-4-6":  {"input":  3.00, "output": 15.00, "cache_read": 0.30},
    "claude-haiku-4-5":   {"input":  0.80, "output":  4.00, "cache_read": 0.08},
    "gpt-4o":             {"input":  2.50, "output": 10.00, "cache_read": 1.25},
    "gpt-4o-mini":        {"input":  0.15, "output":  0.60, "cache_read": 0.075},
}

def cost(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float:
    p = PRICES[model]
    return (
        (input_tokens - cache_read_tokens) / 1_000_000 * p["input"]
        + cache_read_tokens / 1_000_000 * p["cache_read"]
        + output_tokens / 1_000_000 * p["output"]
    )

# Example: a customer service bot doing 100 chats/hour, each ~2000 input tokens, 400 output
print("Customer service bot, 100 chats/hour:")
for model in ["claude-haiku-4-5", "claude-sonnet-4-6", "gpt-4o-mini"]:
    no_cache  = cost(model, 2000, 400) * 100
    w_cache   = cost(model, 2000, 400, cache_read_tokens=1800) * 100
    print(f"  {model:25s}  no-cache: ${no_cache:6.2f}/hr   with-cache: ${w_cache:6.2f}/hr")
```

Run it. This is your interview cheat sheet. When someone asks "what would it cost
to serve 1000 of these queries a day?" you have an answer in 30 seconds. The
prices above are placeholders and drift; verify against the pricing page before
you state a number to anyone.

**Reading for Day 7 (3-4 hours, with coffee):**

1. Anthropic - Prompt Engineering Overview: https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview and all sub-pages
2. Anthropic - Tool Use: https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview
3. OpenAI - Function Calling: https://platform.openai.com/docs/guides/function-calling
4. Simon Willison's blog, last 10 posts: https://simonwillison.net - skim, bookmark
5. Anthropic Cookbook: https://github.com/anthropics/anthropic-cookbook - clone it, skim the structure, run one notebook

```bash
git add cost_calc.py
git commit -m "Day 7: cost calculator + reading-day completion"
git push
```

### Week 1 Wrap-up Checklist

- [ ] 8 scripts committed and pushed to the `llm-fundamentals` repo
- [ ] 40+ API calls made (check Anthropic console "Usage" tab)
- [ ] You can write a basic Anthropic call from memory
- [ ] You can write a basic OpenAI call from memory
- [ ] You can explain temperature in one sentence
- [ ] You can explain the tool-use loop in three sentences without looking
- [ ] You can explain prompt caching in two sentences
- [ ] You've read the four core doc pages above
- [ ] You can read your Anthropic console and find usage, cost, recent activity

If any are "no," close them before Week 2. Do not move forward with gaps.

---

## WEEK 2: RAG From Scratch

Goal: one small RAG system end to end, no framework. The architecture is the
same one your `drupal-rag-cli` used. What changes is ingestion, because your
source is no longer clean scraped HTML. It is PDFs and Word files with running
headers, page numbers, tables, and possibly scanned pages, and it carries a
clause structure worth preserving.

### The one-paragraph diff (vs a clean-HTML RAG build)

| Day | Clean-HTML build | SQF build |
|---|---|---|
| 9 source material | scrape HTML to `data/raw/*.txt` | `ingest.py`: extract PDF/DOCX from `data/source/`, OCR fallback, read tracked-change insertions, strip boilerplate, write page-aware `data/raw/*.jsonl` |
| 10 chunk | fixed 2000-char sliding window | clause-aware chunking on SQF clause numbers (two header forms, filename fallback, cross-page threading), with rich metadata |
| 11 embed | Voyage -> Chroma, metadata `{source, chunk_index}` | same, but wider None-safe metadata, collection `sqf_docs` |
| 12 ask | generic system prompt, `[Excerpt from source]` | compliance system prompt (cite clause, refuse when absent), excerpt header carries clause + page |

---

### Day 8 (Mon): RAG Concepts (Reading + Whiteboard, Light Coding)

**Concept 1: What RAG is and why it exists**

Retrieval-Augmented Generation. You retrieve relevant context from your own data,
augment the LLM prompt with it, and the LLM generates an answer using it. Three
reasons it exists: context-window limits (you cannot stuff a 500-page manual into
every request), freshness (the model's training has a cutoff; your data is
current), and grounding (models hallucinate, and forcing them to answer from
retrieved text reduces that). The unsexy truth: most "AI for business" projects
in 2026 are RAG projects with different paint. Becoming fluent in RAG is the
single highest-leverage technical skill for your pivot.

**Concept 2: Embeddings**

An embedding is a list of numbers (typically 768, 1024, or 1536) representing the
meaning of a piece of text. Similar meaning produces vectors that sit close
together; different meaning, far apart. The contract is all you need: text in,
vector out, similar text produces similar vectors. This lets you search by
meaning instead of keyword.

**Concept 3: Vector databases**

A vector database is a fast lookup for "given this query embedding, find the K
closest stored embeddings." Everything else (filtering, metadata, persistence) is
convenience. Chroma, Pinecone, pgvector, Weaviate, Qdrant, Milvus are variants on
the same core operation. For learning, Chroma is right: local, no setup,
Python-native. You graduate to pgvector later because Postgres is what enterprise
customers actually run.

**Concept 4: Chunking**

You cannot embed a 500-page manual as one vector; it collapses the meaning to
mush. Break documents into chunks, embed each, store them. Two parameters that
come up in interviews: chunk size (too small lacks context, too big gets noisy;
500-800 tokens is a common prose sweet spot) and overlap (adjacent chunks share
some text so meaning is not lost at boundaries). For this corpus you will not use
a fixed size at all; you will chunk on clause boundaries, which is the SQF-
specific move covered on Day 10.

**Concept 5: The full RAG loop**

```
[One-time ingestion]
  documents -> chunks -> embeddings -> vector store

[Per query]
  user question -> embedding -> top-K nearest chunks -> stuff into prompt -> LLM -> answer
```

Five operations. Everything else (reranking, hybrid search, query rewriting) is
optimization on top.

**Day 8 activity: draw it.** On paper, draw the diagram, then a detailed version
with file names and library names. For this build, annotate the ingestion arrow
with "extract + OCR + de-boilerplate + clause-split," because that arrow is where
all your new work lives. Tape it above your monitor. No coding today.

---

### Day 9 (Tue): Source material - `ingest.py`

Your `cert-rag-cli` project already exists with uv and git. Add the Week 2
dependencies and set up the corpus folder. Note there is nothing to scrape, so no
beautifulsoup4/requests this time; instead you extract from local files.

```bash
cd ~/projects/cert-rag-cli
uv add anthropic chromadb voyageai rich python-dotenv
uv add pymupdf python-docx        # PDF + DOCX extraction
uv add ocrmypdf                    # OCR fallback for scanned PDFs

# ocrmypdf needs system binaries; on WSL2/Ubuntu:
sudo apt update && sudo apt install -y tesseract-ocr ghostscript qpdf

mkdir -p data/source
# Drop your SQF PDFs and .docx files into data/source/
```

Set up credentials the way the rest of the project reads them: a gitignored
`.env` at the repo root, loaded once by a tiny `env` module. Get a Voyage key
(voyageai.com), and reuse your Anthropic key:

```bash
cat > .env.example <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
# LANGFUSE_* keys are added in Week 4
EOF
cp .env.example .env    # then edit .env with your real keys
```

Create `env.py` at the repo root. Every module that reads a key imports it
(directly, or through `tracing` in Week 3), so `.env` is loaded before the first
`os.environ` read from a single place:

```python
"""Load the project's .env before anything reads os.environ.

`import env` is the whole contract. Real environment variables always win
(override=False), so CI and production that set them properly are unaffected by
a stray .env on disk. A missing .env is not an error - exported variables alone
remain a valid way to run.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
```

Update `.gitignore` so you never commit the documents, credentials, or the
regenerable index:

```bash
printf '%s\n' '__pycache__/' '.env' 'data/source/' 'data/raw/' '.chroma/' > .gitignore
```

`data/source/` is gitignored deliberately. Internal SQF documents are not yours
to publish, and keeping them out of git is also the correct posture for the
portfolio story: you built a system that runs on private compliance documents
without leaking them.

Now the extractor. There is nothing to scrape, so you extract text from the files
in `data/source/`, clean them, and write page-aware records that Day 10 will
chunk. Four sub-problems in order: extraction, scanned-page fallback, tracked
changes in the Word files, and boilerplate removal.

The `MANIFEST` below is this project's real one, and it is worth reading rather
than skimming: it is the only place the corpus declares what each document *is*.
Two things in it are decisions, not bookkeeping. First, the edition. The Spices,
Inc. certification report identifies this site as certified against **SQF
Fundamentals Edition 1.1 (FSC 19)** - not SQF Code Edition 9, which is the
edition most public SQF writing talks about. Getting this wrong poisons
everything downstream, because the golden set's clause numbers and every answer's
citation are read against a specific standard. Second, the full SQF Code Edition
10 is in the corpus as *reference* and carries its own `edition` tag, so a
retrieved Code 10 clause is never silently presented as a Fundamentals
requirement.

The rest of the manifest is the actual document set: the certification report,
the site's Food Safety Management System manual, the two published standards, and
the `2.x` / `11.x` SOP `.docx` files whose filenames name the clause they
implement (Day 10 leans on that).

Create `ingest.py` in the project root:

```python
"""Day 9 (SQF): Extract PDFs and Word files into page-aware cleaned records.

Output: one JSONL per source document in data/raw/, each line a {page, text}
record. DOCX has no reliable pages, so its page is null. Day 10 (chunk.py)
consumes these.
"""
import json
import re
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
from docx import Document as DocxDocument
from docx.oxml.ns import qn

SOURCE_DIR = Path("data/source")
RAW_DIR = Path("data/raw")

# doc_type and edition are not guessable from the file, so declare them here.
# edition matches YOUR documents: the Spices, Inc. certification report identifies
# this corpus as SQF Fundamentals Edition 1.1 (FSC 19), not SQF Code Edition 9.
_ED = "SQF Fundamentals 1.1"
_ED_CODE10 = "SQF Code 10"   # the full Food Safety Code, a different (newer) standard
MANIFEST = {
    "Certification Report 1.pdf":                     {"doc_type": "report",  "edition": _ED},
    "_Food Safety Management System.pdf":             {"doc_type": "manual",  "edition": _ED},
    # The authoritative published standards themselves (not site SOPs). The site is
    # certified against Fundamentals 1.1; Code Edition 10 is included as reference and
    # kept on its own edition tag so answers do not conflate the two standards.
    "sqf-fundamentals-for-manufacturing-intermediate-09262019-ed-1-1-final.pdf":
                                                      {"doc_type": "standard", "edition": _ED},
    "SQFI-Food-Safety-Code-Edition-10_FSC-19.pdf":    {"doc_type": "standard", "edition": _ED_CODE10},
    "2.1.1 Food Safety Policy.docx":                  {"doc_type": "policy",  "edition": _ED},
    "Food Safety Reporting Structure Statement.docx": {"doc_type": "policy",  "edition": _ED},
    "2.1.2 Management Responsibility.docx":                          {"doc_type": "sop", "edition": _ED},
    "2.1.3 Management Review.docx":                                  {"doc_type": "sop", "edition": _ED},
    "2.1.4 Complaint Management.docx":                               {"doc_type": "sop", "edition": _ED},
    "2.2.1 Food Safety Management System.docx":                      {"doc_type": "sop", "edition": _ED},
    "2.2.2 Document Control.docx":                                   {"doc_type": "sop", "edition": _ED},
    "2.2.3 Records.docx":                                            {"doc_type": "sop", "edition": _ED},
    "2.3.2 Raw and Packaging Materials.docx":                        {"doc_type": "sop", "edition": _ED},
    "2.3.5 Finished Product.docx":                                   {"doc_type": "sop", "edition": _ED},
    "2.4.1 Food Legislation.docx":                                   {"doc_type": "sop", "edition": _ED},
    "2.4.2 Good Manufacturing Practices.docx":                       {"doc_type": "sop", "edition": _ED},
    "2.4.4 Approved Supplier Program.docx":                          {"doc_type": "sop", "edition": _ED},
    "2.4.5 Non-Conforming Material and Product.docx":               {"doc_type": "sop", "edition": _ED},
    "2.4.6 Product Rework.docx":                                     {"doc_type": "sop", "edition": _ED},
    "2.4.7 Product Release.docx":                                    {"doc_type": "sop", "edition": _ED},
    "2.4.8 Environmental Monitoring.docx":                           {"doc_type": "sop", "edition": _ED},
    "2.5.1, 2.5.2 Validation, Effectiveness and Verification.docx":  {"doc_type": "sop", "edition": _ED},
    "2.5.3 Corrective and Preventative Action.docx":                 {"doc_type": "sop", "edition": _ED},
    "2.5.4 Product Sampling, Inspection and Analysis.docx":          {"doc_type": "sop", "edition": _ED},
    "2.5.5 Internal Audits and Inspections.docx":                    {"doc_type": "sop", "edition": _ED},
    "2.6.1, 2.6.2 Product Identification and Trace.docx":            {"doc_type": "sop", "edition": _ED},
    "2.6.3 Withdrawal and Recall.docx":                              {"doc_type": "sop", "edition": _ED},
    "2.7.1 Food Defense Plan.docx":                                  {"doc_type": "sop", "edition": _ED},
    "2.8.1 Allergen Management For Food Fundamentals.docx":          {"doc_type": "sop", "edition": _ED},
    "2.9 Training.docx":                                             {"doc_type": "sop", "edition": _ED},
    "11.1.7 Equipment Utensils.docx":                               {"doc_type": "sop", "edition": _ED},
    "11.2.1 Repairs and Maintenance.docx":                           {"doc_type": "sop", "edition": _ED},
    "11.2.3 Calibration.docx":                                       {"doc_type": "sop", "edition": _ED},
    "11.2.5 Cleaning and Sanitation.docx":                           {"doc_type": "sop", "edition": _ED},
    "11.3.1 Personnel Hygiene and Welfare.docx":                     {"doc_type": "sop", "edition": _ED},
    "11.3.1 Risk Assessment – Personal Items, Jewelry, Electronics and Drink Containers.docx": {"doc_type": "sop", "edition": _ED},
    "11.4.1 Sensory Evaluation Procedure.docx":                      {"doc_type": "sop", "edition": _ED},
    "11.5.1 Water Ice and Air Supply.docx":                          {"doc_type": "sop", "edition": _ED},
    "11.6.1 Receipt, Storage and Handling of Goods.docx":            {"doc_type": "sop", "edition": _ED},
    "11.6.5 Loading Transport and Unloading Practices.docx":         {"doc_type": "sop", "edition": _ED},
    "11.7.3 Foreign Material Control & Detection.docx":              {"doc_type": "sop", "edition": _ED},
    "11.8.1.1 Waste Disposal.docx":                                  {"doc_type": "sop", "edition": _ED},
    "11.8.1.6 Controlled Disposal of Trademarked Materials.docx":    {"doc_type": "sop", "edition": _ED},
    "Process Narratives copy.docx":                                  {"doc_type": "sop", "edition": _ED},
}


def load_pdf(path: Path) -> list[dict]:
    doc = fitz.open(path)
    try:
        return [{"page": i, "text": pg.get_text("text")}
                for i, pg in enumerate(doc, start=1)]
    finally:
        doc.close()


def is_probably_scanned(pages: list[dict], min_chars_per_page: int = 50) -> bool:
    if not pages:
        return False
    return sum(len(p["text"].strip()) for p in pages) / len(pages) < min_chars_per_page


def ensure_text_layer(path: Path) -> Path:
    """OCR a scanned PDF into a searchable copy; return the path to actually read."""
    if not is_probably_scanned(load_pdf(path)):
        return path
    import ocrmypdf
    out = RAW_DIR / f"{path.stem}.ocr.pdf"
    print(f"  {path.name} looks scanned - running OCR")
    ocrmypdf.ocr(path, out, skip_text=True, progress_bar=False)
    return out


def _element_text(element) -> str:
    """Text of a w:p/w:tc in document order, INCLUDING tracked insertions.

    python-docx's Paragraph.text only reads <w:r> runs that are direct children
    of the paragraph, so runs nested in <w:ins> (unaccepted tracked-change
    insertions) are silently dropped - that is how the entire "Recall vs.
    Withdrawal Determination" section of 2.6.3 went missing. We walk every <w:t>
    descendant instead: this picks up <w:ins> text and naturally excludes
    deletions, since deleted text lives in <w:delText>, not <w:t>.
    """
    parts = []
    for node in element.iter():
        if node.tag == qn("w:t"):
            parts.append(node.text or "")
        elif node.tag == qn("w:tab"):
            parts.append("\t")
        elif node.tag in (qn("w:br"), qn("w:cr")):
            parts.append("\n")
    return "".join(parts)


def load_docx(path: Path) -> list[dict]:
    doc = DocxDocument(path)
    parts = []
    for p in doc.paragraphs:
        text = _element_text(p._p).strip()
        if text:
            parts.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [t for c in row.cells if (t := _element_text(c._tc).strip())]
            if cells:
                parts.append(" | ".join(cells))   # keep tables as pipe-joined rows
    return [{"page": None, "text": "\n".join(parts)}]


_PAGE_NUM = re.compile(r"^\s*(page\s+)?\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE)


def strip_boilerplate(pages: list[dict], threshold: float = 0.6) -> list[dict]:
    """Drop lines that repeat on >= threshold of pages (running headers/footers)."""
    counts: Counter[str] = Counter()
    for p in pages:
        for line in {ln.strip() for ln in p["text"].splitlines() if ln.strip()}:
            counts[line] += 1
    n = max(len(pages), 1)
    boiler = {line for line, c in counts.items() if c / n >= threshold}
    cleaned = []
    for p in pages:
        kept = [ln for ln in p["text"].splitlines()
                if ln.strip() not in boiler and not _PAGE_NUM.match(ln)]
        cleaned.append({**p, "text": "\n".join(kept)})
    return cleaned


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(SOURCE_DIR.iterdir()):
        meta = MANIFEST.get(path.name)
        if meta is None:
            print(f"  SKIP {path.name} (not in MANIFEST)")
            continue

        if path.suffix.lower() == ".pdf":
            pages = strip_boilerplate(load_pdf(ensure_text_layer(path)))
        elif path.suffix.lower() in {".docx", ".doc"}:
            pages = load_docx(path)
        else:
            continue

        out = RAW_DIR / f"{path.stem}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for pg in pages:
                f.write(json.dumps({**pg, "source": path.name, **meta}) + "\n")
        print(f"  {path.name} -> {out.name} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
```

Run it and spot-check:

```bash
uv run python ingest.py
# Certification Report 1.pdf -> Certification Report 1.jsonl (43 pages)
# sqf-fundamentals-...-ed-1-1-final.pdf -> ...jsonl (56 pages)
# SQFI-Food-Safety-Code-Edition-10_FSC-19.pdf -> ...jsonl (74 pages)
# 2.5.5 Internal Audits and Inspections.docx -> ...jsonl (1 pages)
# ... 44 files in all

head -1 "data/raw/Certification Report 1.jsonl" | python -m json.tool
```

You want real clause text, not a wall of repeated header lines. If a known
scanned form produced empty pages, confirm the OCR branch fired (you would have
seen the "looks scanned" line). If boilerplate persists, adjust `threshold`;
version footers that appear on every page get caught at 0.6.

Spot-check a `.docx` that you know has unaccepted tracked changes, and grep the
raw output for a phrase that only appears inside an insertion. This is the check
that caught the `_element_text` bug: `2.6.3 Withdrawal and Recall.docx` renders
fine in Word, but its "Recall vs. Withdrawal Determination" section lives
entirely inside `<w:ins>` runs, so `paragraph.text` returned a document missing a
section - with no error and no empty page to notice. A silent partial extraction
is the worst failure mode in this whole pipeline: the answer still looks
confident, it is just answering from a document that lost a requirement.

```bash
git add ingest.py .gitignore pyproject.toml
git commit -m "Day 9: PDF/DOCX ingestion with OCR fallback and boilerplate removal"
```

---

### Day 10 (Wed): `chunk.py` - clause-aware chunking

A fixed-size window is wrong here. SQF documents are organized as numbered
clauses (`2.4.3.1 Internal Audits`), and a clause is the natural unit of an
auditable requirement. Chunk on clause boundaries and carry the clause number,
title, and page as metadata. That metadata is what later turns a generic answer
into an audit-grade citation.

Create `chunk.py`:

```python
"""Clause-aware chunking.

Reads the page-aware records from ingest.py (data/raw/*.jsonl) and emits one
chunk per clause to data/chunks.jsonl, carrying clause metadata. Same output
contract as a plain chunker (a jsonl of {id, source, text, ...}), just richer
and split on clauses instead of a fixed window.

The clause number is load-bearing: it is what the model cites and what the eval
scores retrieval against, so a chunk that loses it is unciteable and can never
count as a hit. Three things have to go right for it to survive, and each was
dropping a different slice of the corpus:

  - the published standards put the clause number on a line of its own with the
    requirement text below it, so a pattern that expects "2.5.5 Internal Audits"
    on one line matches nothing in them;
  - a requirement that spans a page break continues on a page whose first line
    is not a header, so splitting page-by-page orphans the continuation;
  - the site SOPs carry no clause number in their body at all - it is in the
    filename ("2.8.1 Allergen Management For Food Fundamentals.docx").
"""
import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/chunks.jsonl")

MAX_CHARS = 2400        # sub-split a clause only if it exceeds this (~600 tokens)
HEADING_SEGMENTS = 3    # 2.1.2 is a heading with a title; 2.1.2.4 is a requirement under it
MAX_TITLE_CHARS = 80

# Two header forms. Inline ("2.4.3.1 Internal Audits") is what the SOPs and the
# audit report use; standalone is what both published standards use. Both
# require at least one dot so a plain list number ("1.") does not false-trigger.
_CLAUSE_INLINE = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s+(\S.*)$")
_CLAUSE_ALONE = re.compile(r"^\s*(\d+(?:\.\d+){1,4})\s*$")

# Table-of-contents rows ("Food Safety Policy ......... 24"). Left in, every ToC
# entry becomes a contentless clause chunk competing with the real requirement.
_TOC_LEADER = re.compile(r"\.{4,}")

# "i.", "ii.", "a)", "3." - list markers pymupdf emits on their own line.
_LIST_MARKER = re.compile(r"^\s*(?:[ivxlc]+|[a-z]|\d{1,2})[.)]\s*$", re.IGNORECASE)

# Leading clause number(s) in a filename stem, e.g. "2.5.1, 2.5.2 Validation...".
_FILENAME_CLAUSE = re.compile(r"^(\d+(?:\.\d+){1,4})")
_FILENAME_PREFIX = re.compile(r"^(?:\d+(?:\.\d+){1,4}[,\s]*)+")


def _looks_like_title(line: str) -> bool:
    """True if this line reads as a clause title rather than requirement prose.

    A title is short and self-contained; requirement text runs long and breaks
    mid-sentence, so it ends on a comma, colon or wrapped word.

    A clause number is explicitly not a title. Stripping ToC leader lines leaves
    each ToC entry's number sitting directly above the next one, and without this
    guard "2.1.2" adopts "2.1.3" as its title and every requirement beneath it
    inherits that instead of "Management Responsibility".
    """
    s = line.strip()
    if not s or len(s) > MAX_TITLE_CHARS:
        return False
    if _CLAUSE_ALONE.match(s) or _CLAUSE_INLINE.match(s):
        return False
    return s[-1] not in ".,;:" and not _LIST_MARKER.match(s)


def _inherited_title(titles: dict[str, str], clause: str) -> str | None:
    """Title of the nearest ancestor heading - 2.1.2.4 inherits from 2.1.2.

    Requirement clauses in the standards have no title of their own, so without
    this they carry clause_title=None and the citation header shows a bare number.
    """
    parts = clause.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:cut])
        if parent in titles:
            return titles[parent]
    return None


def filename_clause(source: str) -> tuple[str | None, str | None]:
    """(clause, title) named by a source filename, or (None, None).

    Used only as a fallback for documents whose body text carries no clause
    header - the SOPs are written PURPOSE / SCOPE / PROCEDURE with the clause
    they implement only in the filename.
    """
    stem = Path(source).stem
    m = _FILENAME_CLAUSE.match(stem)
    if not m:
        return None, None
    return m.group(1), _FILENAME_PREFIX.sub("", stem).strip() or None


def document_lines(recs: list[dict]):
    """Yield (page, line) for every line of a document, in reading order.

    Iterating the whole document rather than each page in isolation is what
    carries clause context across a page break.
    """
    for rec in recs:
        for line in rec["text"].splitlines():
            if not _TOC_LEADER.search(line):
                yield rec["page"], line


def split_into_clauses(lines):
    """Split a document's (page, line) stream into per-clause chunks.

    Yields dicts of {clause, clause_title, page, text, from_header}. from_header
    distinguishes a real clause header from preamble text, so contentless
    headings can be dropped without dropping unnumbered prose.
    """
    lines = list(lines)
    titles: dict[str, str] = {}
    chunks: list[dict] = []
    current: dict | None = None

    def start(clause, title, page, header_line, from_header):
        nonlocal current
        if current is not None:
            chunks.append(current)
        if clause and title:
            titles.setdefault(clause, title)
        current = {
            "clause": clause,
            "clause_title": title,
            "page": page,
            "body": [header_line] if header_line else [],
            "from_header": from_header,
        }

    i = 0
    while i < len(lines):
        page, line = lines[i]
        alone = _CLAUSE_ALONE.match(line)
        inline = _CLAUSE_INLINE.match(line)

        if alone:
            clause = alone.group(1)
            title, consumed = None, 1
            # Only heading-level numbers take the next line as a title; for a
            # requirement number that next line is the requirement itself.
            if (clause.count(".") + 1 <= HEADING_SEGMENTS
                    and i + 1 < len(lines) and _looks_like_title(lines[i + 1][1])):
                title, consumed = lines[i + 1][1].strip(), 2
            resolved = title or _inherited_title(titles, clause)
            header = f"{clause} {resolved}" if resolved else clause
            start(clause, resolved, page, header, True)
            i += consumed
            continue

        if inline:
            clause, title = inline.group(1), inline.group(2).strip()
            if not _looks_like_title(title):
                title = _inherited_title(titles, clause)
            start(clause, title, page, line.strip(), True)
            i += 1
            continue

        # Text with no clause header above it. Threading context across pages is
        # what preserves a clause number through a page break, but unnumbered
        # prose has no context worth preserving - and letting it run on would
        # merge a document's whole front matter into one chunk stamped with the
        # first page. Break those at the page boundary so the cited page is right.
        if current is None or (current["clause"] is None and page != current["page"]):
            start(None, None, page, line, False)
        else:
            current["body"].append(line)
        i += 1

    if current is not None:
        chunks.append(current)

    for ch in chunks:
        body = [ln for ln in ch["body"] if ln.strip()]
        # A clause header with nothing under it is a heading or a ToC remnant -
        # no requirement text, so nothing worth retrieving.
        if ch["from_header"] and len(body) <= 1:
            continue
        text = "\n".join(body).strip()
        if text:
            yield {**ch, "text": text}


def sub_split(body: str, size: int):
    """Only used when a single clause is very long. Keeps clause metadata intact."""
    if len(body) <= size:
        yield body
        return
    start = 0
    while start < len(body):
        yield body[start:start + size]
        start += size - 200   # small overlap


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    stats: Counter[str] = Counter()
    # Same clause number can appear more than once in a document (e.g. audit
    # reports cite a clause repeatedly), so disambiguate repeated base ids.
    seen_ids: dict[str, int] = {}
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for jf in sorted(RAW_DIR.glob("*.jsonl")):
            recs = [json.loads(line) for line in jf.open(encoding="utf-8")]
            if not recs:
                continue
            source = recs[0]["source"]
            fb_clause, fb_title = filename_clause(source)

            for ch in split_into_clauses(document_lines(recs)):
                clause, title = ch["clause"], ch["clause_title"]
                clause_source = "header" if clause else None
                if clause is None and fb_clause:
                    clause, clause_source = fb_clause, "filename"
                    title = title or fb_title

                for i, piece in enumerate(sub_split(ch["text"], MAX_CHARS)):
                    if not piece.strip():
                        continue
                    # Counted per written chunk, not per clause, so the reported
                    # coverage matches what actually lands in the corpus.
                    stats[clause_source or "none"] += 1
                    module = clause.split(".")[0] if clause else None
                    base_id = f"{Path(source).stem}__{clause or 'preamble'}__{ch['page']}__{i}"
                    n = seen_ids.get(base_id, 0)
                    seen_ids[base_id] = n + 1
                    chunk_id = base_id if n == 0 else f"{base_id}__{n}"
                    out.write(json.dumps({
                        "id": chunk_id,
                        "text": piece,
                        "source": source,
                        "doc_type": recs[0].get("doc_type"),
                        "edition": recs[0].get("edition"),
                        "module": module,
                        "clause": clause,
                        # How the clause was determined: a header in the text, or
                        # the filename. Keeps the weaker signal auditable.
                        "clause_source": clause_source,
                        "clause_title": title,
                        "page": ch["page"],
                    }) + "\n")
                    written += 1

    total = sum(stats.values())
    unclaused = stats["none"]
    print(f"Wrote {written} clause chunks to {OUT_FILE}")
    print(f"  clause from header:   {stats['header']}")
    print(f"  clause from filename: {stats['filename']}")
    print(f"  no clause:            {unclaused} ({100 * unclaused / max(total, 1):.0f}%)")


if __name__ == "__main__":
    main()
```

```bash
uv run python chunk.py
# Wrote 971 clause chunks to data/chunks.jsonl
#   clause from header:   816
#   clause from filename: 87
#   no clause:            68 (7%)
head -1 data/chunks.jsonl | python -m json.tool
```

That closing coverage report is the point of the whole exercise, and it is worth
printing every run. A chunk with no clause is not junk - front matter and process
narratives legitimately have none - but it can never be a retrieval *hit*,
because the eval scores retrieval by clause. If that last number climbs after you
change a regex, you just made part of the corpus unciteable, and nothing else in
the pipeline will tell you.

Two deliberate rules survive from the simple version: do not merge across clause
boundaries to hit a size floor (blurring two requirements into one chunk is the
exact failure you are avoiding), and only sub-split a clause when it is genuinely
long, copying the clause metadata onto every sub-piece so citation survives.

Everything else here exists because a specific slice of this corpus was being
dropped:

- **Two header forms.** `_CLAUSE_INLINE` matches `2.4.3.1 Internal Audits` (the
  SOPs and the audit report). Both *published standards* instead put a bare
  `2.5.5` on its own line with the requirement below it, which the inline
  pattern matches zero times - so `_CLAUSE_ALONE` handles that, taking the next
  line as a title only when the number is heading-level (`HEADING_SEGMENTS`).
  For a deeper requirement number the next line is the requirement itself, not a
  title.
- **Filename fallback.** The site SOPs are written PURPOSE / SCOPE / PROCEDURE
  and never state their clause in the body; it is in the filename.
  `filename_clause()` recovers it, and `clause_source` (`header` | `filename` |
  `null`) records which signal was used so the weaker one stays auditable. That
  fallback is 87 of the 971 chunks - roughly a tenth of the corpus that a
  body-text-only chunker leaves unciteable.
- **Cross-page threading.** `document_lines()` iterates the whole document, not
  page by page. A requirement that spans a page break continues on a page whose
  first line is not a header, so page-at-a-time chunking orphans the
  continuation from its clause number.
- **ToC stripping and title inheritance.** `_TOC_LEADER` drops
  `Food Safety Policy ......... 24` rows, which would otherwise each become a
  contentless clause chunk competing with the real requirement.
  `_looks_like_title` and `_inherited_title` then let `2.1.2.4` inherit
  "Management Responsibility" from `2.1.2`, so a sub-clause with no title of its
  own still renders a meaningful citation header instead of a bare number.
- **Repeated ids.** An audit report cites the same clause on several pages, so
  `seen_ids` disambiguates a repeated base id rather than letting the second
  chunk overwrite the first at embed time.

If the clause regexes miss your documents' numbering, tune them against a real
page before moving on, and watch the coverage line rather than the total. This is
the one heuristic worth getting right by inspection.

```bash
git add chunk.py
git commit -m "Day 10: clause-aware chunker with clause/page metadata"
```

---

### Day 11 (Thu): `embed.py` - Voyage embeddings into Chroma

This mirrors a standard embed step, with three SQF touches: rename the
collection, carry the new metadata into Chroma, and strip `None` values because
Chroma rejects them.

```python
"""Day 11: Embed chunks and store in a local Chroma collection."""
import json
import os
import time
from pathlib import Path

import chromadb
import voyageai
from rich.progress import track

# Loads .env so VOYAGE_API_KEY below resolves the same way it does for the query
# path. This script never imports tracing, so it needs its own env import.
import env  # noqa: F401

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "sqf_docs"
EMBED_MODEL = "voyage-3-lite"   # cheap, good enough for learning; upgrade to voyage-3 later


# Chroma metadata values must be str/int/float/bool - never None. Drop None keys.
# clause_source records how the clause was determined (header vs filename); see
# Day 10, where the chunker adds it.
def _clean_meta(rec: dict) -> dict:
    fields = ("source", "doc_type", "edition", "module", "clause", "clause_source",
              "clause_title", "page")
    return {k: rec[k] for k in fields if rec.get(k) is not None}


def main():
    voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    records = [json.loads(line) for line in CHUNKS_FILE.open()]
    print(f"Embedding {len(records)} chunks...")

    # Free tier: 3 RPM, 10K TPM -> small batches + sleep.
    BATCH = 8
    SLEEP_BETWEEN_BATCHES = 21
    batches = list(range(0, len(records), BATCH))
    for idx, i in enumerate(track(batches, description="Embedding")):
        batch = records[i:i + BATCH]
        texts = [r["text"] for r in batch]

        result = voyage.embed(texts=texts, model=EMBED_MODEL, input_type="document")

        collection.add(
            ids=[r["id"] for r in batch],
            embeddings=result.embeddings,
            documents=texts,
            metadatas=[_clean_meta(r) for r in batch],
        )
        if idx < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Stored {collection.count()} vectors in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
```

```bash
uv run python embed.py
# Embedding 971 chunks...
# Stored 971 vectors in collection 'sqf_docs'.
```

At `BATCH = 8` and a 21-second sleep between batches, 971 chunks is ~122 batches
and about **45 minutes**. That is expected, and it is why this script rebuilds
the collection from scratch rather than trying to be incremental - but also why
you should not casually re-run it. Get `chunk.py` right first; embedding is the
slow, expensive end of the pipeline.

```bash
git add embed.py
git commit -m "Day 11: embed to sqf_docs with clause metadata (None-safe)"
```

---

### Day 12 (Fri): `ask.py` - retrieve, cite clauses, refuse when absent

The answer path is a standard vanilla-retrieval RAG loop with tracing. Two SQF
edits: the system prompt (in a compliance context a fabricated requirement is
worse than a refusal, and every claim carries a clause) and `assemble_prompt` (so
the clause and page reach the model). The query-embedding helper and the vanilla
retriever come next; Days in Week 4 add hybrid and rerank behind the same
interface.

`retrievers/embed.py` (query embedding, shared by every strategy):

```python
"""Query embedding via Voyage."""
import voyageai

EMBED_MODEL = "voyage-3-lite"

def embed_query(query: str) -> list[float]:
    voyage = voyageai.Client()
    return voyage.embed([query], model=EMBED_MODEL, input_type="query").embeddings[0]
```

`retrievers/vanilla.py` (note it returns clause and page so citations work):

```python
"""Vanilla cosine-similarity retrieval."""
import chromadb

from retrievers.embed import embed_query

CHROMA_DIR = ".chroma"
COLLECTION_NAME = "sqf_docs"

def retrieve(query: str, k: int = 5):
    chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma.get_collection(COLLECTION_NAME)
    q_emb = embed_query(query)
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    return [
        {
            "text": doc,
            "source": meta["source"],
            "clause": meta.get("clause"),
            "clause_title": meta.get("clause_title"),
            "page": meta.get("page"),
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
```

`ask.py`:

```python
"""Day 12: Ask a question. Retrieve context. Generate an answer."""
import os
import sys

from anthropic import Anthropic

from retrievers.vanilla import retrieve

# 14, not 5: clause chunks are small (~690 chars), and at k=5 completeness
# was the weakest judged axis. A 5/10/14 sweep moved it 3.85 -> 4.24 with
# clause citation flat at ~97%. Env-overridable so the eval can sweep it.
TOP_K = int(os.getenv("TOP_K", "14"))
LLM_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You answer questions about the SQF certification documents in
the provided excerpts, for a food-safety practitioner.

Rules:
- Answer only from the excerpts. If they do not contain the requirement, say
  "Not found in the provided documents" and stop. Never supply a requirement
  from general knowledge or another standard.
- Cite the clause for every requirement you state, as (source, clause, p.page).
  If an excerpt has no clause number, cite the source and page.
- If an answer spans multiple clauses, list each with its own citation.
- Use plain hyphens, never em dashes."""


def assemble_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[Excerpt {i} | {c['source']}"
        if c.get("clause"):
            header += f" | clause {c['clause']}"
        if c.get("page") is not None:
            header += f" | p.{c['page']}"
        header += "]"
        blocks.append(f"{header}\n{c['text']}")
    context = "\n\n---\n\n".join(blocks)
    return f"""Documentation excerpts:

{context}

---

Question: {query}

Answer using only the excerpts above, citing clauses."""


def answer_question(query: str) -> str:
    chunks = retrieve(query, k=TOP_K)
    prompt = assemble_prompt(query, chunks)
    anthropic = Anthropic()
    response = anthropic.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python ask.py "your question"')
        sys.exit(1)
    print(answer_question(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
```

Test the two behaviors that matter:

```bash
uv run python ask.py "How often must internal audits be conducted?"
# Expect an answer citing a real clause, e.g.
# (sqf-fundamentals-for-manufacturing-intermediate-09262019-ed-1-1-final.pdf, 2.5.5.1, p.28)

uv run python ask.py "What is the maximum fine for an OSHA violation?"
# Expect "Not found in the provided documents" - this is out of corpus
```

The second test is the important one. A confident, wrong answer there is a
failure; a refusal is a pass.

```bash
git add ask.py retrievers/
git commit -m "Day 12: compliance system prompt, clause citations, metadata passthrough"
```

Langfuse tracing is added in Week 4 Day 8-9, at which point `answer_question`,
`retrieve`, and the generate call get `@observe` decorators. The structure above
is deliberately the un-instrumented version so Week 4 has something to wrap.

---

### Day 13 (Sat): Refine and Iterate

Ask ten real questions, read the retrieved chunks, and adjust. The SQF-specific
things to watch:

- Clause splitting: are any chunks straddling two clauses? Fix `_CLAUSE`.
- Tables: did a requirements table survive as readable pipe-joined rows, or did
  extraction scramble it? If a key requirement lives in a mangled table, that is
  the one place to reach for pdfplumber's table extraction on that specific file.
- Refusals: try five questions you know are out of corpus and confirm the system
  refuses all five.

---

### Day 14 (Sun): README, Polish, Publish

Same structure as any RAG README, framed as a compliance/regulatory RAG over SQF
food-safety certification documents. In "Notes on choices," add the two decisions
unique to this corpus: clause-aware chunking (why you split on clauses rather than
a fixed window) and the refusal-first system prompt (why fabrication is the
primary risk in a compliance setting). Keep `data/source/` gitignored; the repo
ships the code and the pipeline, not the documents.

Pipeline-drawing test: you should be able to draw
`data/source/*.pdf,docx -> ingest.py -> data/raw/*.jsonl -> chunk.py ->
data/chunks.jsonl -> embed.py -> Chroma(sqf_docs) -> ask.py` from memory before
you call Week 2 done.

### Week 2 Wrap-up Checklist

- [ ] `ingest.py` extracts PDF and DOCX, OCRs scanned files, reads tracked-change
      insertions, strips boilerplate
- [ ] `chunk.py` splits on clauses and carries clause/page metadata, and its
      no-clause coverage number is one you have looked at and accepted
- [ ] `embed.py` stores None-safe metadata in the `sqf_docs` collection
- [ ] `ask.py` cites clauses and refuses out-of-corpus questions
- [ ] Repo pushed; `data/source/` and `.chroma/` gitignored
- [ ] You can draw the SQF pipeline from memory

---

## A Note on Pacing

Week 1 is deliberately corpus-independent fundamentals; do not let it sprawl. The
SQF-specific work is Day 9 (ingestion), Day 10 (clause chunking), and the Day 12
compliance prompt. If you fall behind, protect those three; they are what makes
this a compliance RAG rather than a generic docs bot.
