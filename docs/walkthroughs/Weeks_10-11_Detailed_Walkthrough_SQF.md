# Weeks 10-11 Detailed Walkthrough: Production & Cloud

**For Rob Snyder. Working tutorial, not just a plan. Aim: by end of Week 11 you've deployed one of your agents to a cloud AI platform (AWS Bedrock or Azure OpenAI), documented the migration the way an SA documents it for a customer, and taken (or scheduled) a cloud AI certification.**

**Why this phase matters**: Enterprise AI dollars flow through cloud platforms, not direct API keys. Customers run on AWS, Azure, or GCP and need their AI to live there too - for procurement, data residency, IAM, audit logging, and cost allocation. An AI Solutions Architect who can't speak cloud deployment is incomplete. This phase makes your portfolio enterprise-credible and gets your resume past recruiter keyword filters with a recognized cert.

This is two weeks (14 days). Week 10 deploys to the cloud. Week 11 is the certification sprint plus polish.

---

## Before You Start: Prerequisites + The One Big Decision (20 minutes)

Coming into Week 10 you should have, from Weeks 1-9:

- [ ] `commerce-ai-ops-agent` repo (raw + LangGraph versions)
- [ ] `cert-rag-cli` repo (SQF compliance RAG: three retrieval strategies, clause-citation + refusal evals, Langfuse)
- [ ] `drupal-mcp-server` published
- [ ] All the foundational skills from the prior phases

### The big decision: AWS Bedrock or Azure OpenAI?

Make this choice on day one and don't look back. The skills transfer; depth on one beats shallow exposure to both.

**Choose AWS Bedrock if**: you want breadth, you're targeting AI infra startups or AWS-aligned companies, or you want the broadest model selection (Claude, Llama, Mistral, Nova, Cohere all on one platform). The certification (AIF-C01) is cheaper and faster.

**Choose Azure OpenAI if**: you're targeting enterprise, regulated industries (healthcare, finance, government), or Microsoft-shop employers. The certification (AI-102) is harder but carries more weight with enterprise buyers.

Given your background - e-commerce, commerce AI vendors, AI infra companies on your target list - **AWS Bedrock is the slightly better default**, mostly because Claude runs natively on Bedrock (so your existing code ports cleanly) and the cert is a faster win. But Azure is a defensible choice if your target companies skew enterprise. Pick one now.

The rest of this walkthrough shows **AWS Bedrock** as the primary path with **Azure OpenAI** notes where the approach differs. Follow your chosen platform's track.

### Cost note

Weeks 10-11 have two cost buckets: cloud platform usage (a few dollars of Bedrock/Azure inference) and the certification exam ($100 for AWS AIF-C01, similar for Azure AI-102). Budget ~$120 total. The cert is the bigger line item and the better investment.

---

# WEEK 10: Deploy to the Cloud

**Goal**: Take your LangGraph agent and make it run with inference routed through your chosen cloud platform instead of the direct Anthropic API. Document everything an SA would document.

---

## Day 1 (Mon): Why Cloud AI Platforms Exist (Concepts + Account Setup)

### Concept: Direct API vs cloud platform

You've been calling `api.anthropic.com` directly with an API key. That's perfect for prototypes and small products. Enterprises rarely do this. They route the same models through their cloud platform (Bedrock, Azure OpenAI, Vertex). Why?

- **Procurement and billing**: one vendor relationship (their existing AWS/Azure contract) instead of a separate Anthropic contract. AI spend shows up on the cloud bill they already manage.
- **Data residency and compliance**: the request stays within their cloud's region and security boundary. Critical for HIPAA, GDPR, FedRAMP.
- **IAM integration**: access controlled by their existing identity system (AWS IAM, Azure Entra), not loose API keys.
- **Audit logging**: every call logged in CloudTrail / Azure Monitor for compliance.
- **Cost allocation**: AI spend tagged and attributed by team/project automatically.

The crucial interview point: **the model is the same.** Claude on Bedrock is the same Claude as the direct API. What changes is auth, region, billing, and the request envelope - not the intelligence. SAs spend a lot of time explaining exactly this to customers.

### Concept: What actually differs in the code

Three things change when you move from direct API to Bedrock:

1. **Auth**: AWS IAM credentials (access key/secret, or a role) instead of an Anthropic API key.
2. **Region**: you pick an AWS region; not all models are in all regions.
3. **Request shape**: Bedrock's `converse()` API has its own envelope (slightly different from the native Messages API), though the messages/content structure is similar. Anthropic's own SDK also offers an `AnthropicBedrock` client that keeps the native shape.

### Project: AWS account + Bedrock model access

**AWS path:**

1. Create or sign into an AWS account at https://aws.amazon.com
2. Install the AWS CLI v2 in WSL2:
   ```bash
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip && sudo ./aws/install
   aws --version
   ```
3. Create an IAM user with programmatic access and the `AmazonBedrockFullAccess` policy (for learning; tighten in production). Generate an access key.
4. Configure credentials:
   ```bash
   aws configure
   # Enter access key, secret, region (us-east-1), output (json)
   ```
5. Request Bedrock model access: AWS Console -> Bedrock -> Model access -> request access to the Anthropic Claude models. Approval is usually instant for Claude.

Verify access:

```bash
aws bedrock list-foundation-models --by-provider anthropic --region us-east-1 --query "modelSummaries[].modelId" --output text
```

You should see a list of available Claude model IDs.

**Azure path (if you chose Azure):**

1. Azure account at https://portal.azure.com
2. Create an Azure OpenAI resource (may require requesting access to the service)
3. Deploy a model (e.g., GPT-4o) to get an endpoint and deployment name
4. Grab the endpoint URL and key from the resource's "Keys and Endpoint" page

No code commit today; account setup.

---

## Day 2 (Tue): First Bedrock Call

### Project: `cloud/bedrock_hello.py`

The Bedrock equivalent of your Week 1 hello. Two approaches - learn both because interviews probe both.

**Approach A: boto3 Converse API** (AWS-native, works for any Bedrock model):

```python
"""Week 10 Day 2: First Bedrock call via boto3 Converse API."""
import boto3
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Use an inference profile ID for current Claude models; check your
# list-foundation-models output for the exact ID available in your region.
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # substitute the current Sonnet ID

conversation = [
    {"role": "user", "content": [{"text": "In one sentence, what is Drupal Commerce?"}]}
]

try:
    response = client.converse(
        modelId=MODEL_ID,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0},
    )
    print(response["output"]["message"]["content"][0]["text"])
    print(f"\nUsage: {response['usage']}")
    print(f"Stop reason: {response['stopReason']}")
except (ClientError, Exception) as e:
    print(f"ERROR: {e}")
```

**Approach B: Anthropic's AnthropicBedrock client** (keeps the native Messages API shape you already know):

```python
"""Same call via the AnthropicBedrock client - native Messages API shape."""
from anthropic import AnthropicBedrock

# Reads AWS credentials from your environment / ~/.aws/credentials
client = AnthropicBedrock(aws_region="us-east-1")

response = client.messages.create(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",  # substitute current ID
    max_tokens=512,
    messages=[{"role": "user", "content": "In one sentence, what is Drupal Commerce?"}],
)
print(response.content[0].text)
```

Install and run:

```bash
cd ~/projects/commerce-ai-ops-agent
uv add boto3
mkdir cloud
# create both files
uv run python cloud/bedrock_hello.py
```

The two approaches return the same answer. Approach A (boto3 Converse) is the AWS-native, model-agnostic way - use it when you might swap Claude for Llama or Nova. Approach B (AnthropicBedrock) keeps your existing Anthropic SDK code nearly unchanged - use it when you're committed to Claude and want minimal code changes. Knowing both, and when to use each, is exactly the kind of nuance an SA interview rewards.

**Azure note**: the equivalent first call uses the `openai` library pointed at your Azure endpoint with `AzureOpenAI(azure_endpoint=..., api_key=..., api_version=...)` and your deployment name as the model. Same messages shape as OpenAI.

Commit:

```bash
git add cloud/
git commit -m "Day 2 (wk10): first Bedrock calls - boto3 Converse and AnthropicBedrock client"
```

---

## Day 3 (Wed): Port the Agent to Bedrock

### Concept: Minimal-change migration

The cleanest migration uses Approach B (AnthropicBedrock) because your agent already speaks the Messages API. You swap the client, change the model ID, and the rest of your orchestration loop is unchanged. That "it just works with a client swap" experience is itself a finding for your writeup: well-structured code ports cleanly.

### Project: `cloud/agent_bedrock.py`

Copy your `agent.py` and change only the client and model:

```python
"""Week 10 Day 3: Commerce agent running on Bedrock (AnthropicBedrock client)."""
import json
from anthropic import AnthropicBedrock

import tools
from guardrail import apply_brand_guardrail

# The only real changes from agent.py: the client and the model ID
client = AnthropicBedrock(aws_region="us-east-1")
MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # substitute current ID

# TOOL_SCHEMAS, TOOL_FUNCTIONS, SYSTEM_PROMPT: identical to agent.py
# ... (import them from agent.py to avoid duplication) ...
from agent import TOOL_SCHEMAS, TOOL_FUNCTIONS, SYSTEM_PROMPT


def run_agent_bedrock(sku: str, max_turns: int = 8) -> str:
    messages = [{"role": "user", "content": f"Produce a merchandising recommendation for SKU {sku}."}]
    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            return apply_brand_guardrail(response.content[0].text)["revised_text"]
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = TOOL_FUNCTIONS[block.name](**block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
            messages.append({"role": "user", "content": results})
    return "Max turns reached."


if __name__ == "__main__":
    import sys
    sku = sys.argv[1] if len(sys.argv) > 1 else "GM-001"
    print(run_agent_bedrock(sku))
```

Run it:

```bash
uv run python cloud/agent_bedrock.py GM-001
```

Same agent, same recommendation, now running through Bedrock. Note any differences you hit: model ID format, region availability, latency, error messages. These notes are your `CLOUD_DEPLOYMENT.md` material.

**Azure note**: porting to Azure OpenAI means swapping to the `AzureOpenAI` client and adjusting the tool-use format to OpenAI's function-calling shape (slightly different from Anthropic's). More code change than the Bedrock+Claude path, which is itself a point worth noting: staying within one model family across platforms is easier than switching families.

Commit:

```bash
git add cloud/agent_bedrock.py
git commit -m "Day 3 (wk10): agent ported to Bedrock via client swap"
```

---

## Day 4 (Thu): IAM, Least Privilege, and Guardrails

### Concept: The security story SAs tell

A customer will ask "how do we control who can use this and what it can do?" The answers live in IAM and platform guardrails. You don't need to be a security engineer, but you need to demonstrate the concepts.

### Project: Tighten IAM to least privilege

Replace the broad `AmazonBedrockFullAccess` with a scoped policy that only allows invoking the specific models you use. Create a policy like:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-*"
    }
  ]
}
```

Attach it, remove the full-access policy, and re-run your agent to confirm it still works with the tighter permissions. This is a small thing that demonstrates a big concept: principle of least privilege, which every enterprise security review cares about.

### Concept: Bedrock Guardrails (platform-level safety)

Bedrock offers a Guardrails feature - platform-managed content filtering, PII redaction, and topic blocking applied at the API boundary, independent of your application code. This is different from your brand-voice guardrail (which is application logic). Read about it (AWS Bedrock Guardrails docs) and write a paragraph for your deployment doc on how you'd layer platform guardrails (safety, PII) under your application guardrails (brand voice). Layered guardrails is a sophisticated SA talking point.

You don't have to implement Bedrock Guardrails fully - understanding the layering is the deliverable.

Commit any policy files / notes:

```bash
git add cloud/iam-policy.json
git commit -m "Day 4 (wk10): least-privilege IAM policy, guardrail layering notes"
```

---

## Day 5 (Fri): Cost, Latency, and Regional Considerations

### Concept: The questions every customer asks

"What will this cost at scale?" and "How fast is it?" You answered the cost question conceptually in Week 1; now answer it with cloud-specific reality.

### Project: Measure and document

Run your agent 10 times on Bedrock and capture from your Langfuse traces (or boto3 response `usage` and `metrics`):

- Average tokens per agent run (input + output across all the model calls in the loop)
- Average cost per run (tokens times Bedrock pricing)
- Average latency per run (Bedrock reports `latencyMs` in `metrics`)
- How many model calls a single agent run makes (the orchestrator loop plus the guardrail)

Build a small table. Then extend it to a scenario: "at 1,000 agent runs per day, monthly cost = X, p50 latency = Y." That projection is exactly what a customer needs and exactly what you'd present in a solution proposal.

### Concept: Regional and model-availability nuance

Document the gotchas you hit:

- Not all Claude models are available in all regions. You may need to use an inference profile ID (cross-region routing) for newer models.
- Bedrock has on-demand pricing and provisioned-throughput pricing (reserved capacity for predictable high volume). Know that provisioned throughput exists and when a customer would want it (steady high traffic, latency guarantees).
- Global vs regional endpoints (newer Claude models on Bedrock offer both).

Write these into your deployment doc. They're the kind of operational detail that signals you've actually deployed, not just read about it.

Commit:

```bash
git add cloud/cost_analysis.md
git commit -m "Day 5 (wk10): cost/latency measurement and regional considerations"
```

---

## Day 6 (Sat): Write CLOUD_DEPLOYMENT.md

### Project: The SA-grade deployment document

This document is portfolio gold - it's the exact artifact a Solutions Architect produces for a customer. Write it well.

```markdown
# Deploying the Commerce AI Ops Agent on AWS Bedrock

## Why Bedrock (vs direct Anthropic API)
- Procurement: AI spend on the existing AWS bill
- Compliance: requests stay in-region within the AWS security boundary
- IAM: access controlled by existing AWS identity, not loose API keys
- Audit: every call in CloudTrail
- Cost allocation: tagged by project/team

## Migration approach
Used the AnthropicBedrock client, which preserves the native Messages API.
The agent's orchestration loop required no changes - only the client
constructor and model ID. Well-structured application code ports cleanly;
the model is identical, only auth/region/billing differ.

## Authentication
IAM user with a least-privilege policy scoped to bedrock:InvokeModel on
the specific Claude model ARNs. (policy included: iam-policy.json)

## Regional considerations
- Model availability varies by region (us-east-1 used here)
- Newer models may require inference profile IDs for cross-region routing
- On-demand vs provisioned-throughput pricing tradeoff

## Cost and latency (measured)
- Avg tokens per agent run: X
- Avg cost per run: $X
- Avg latency per run: X ms
- Model calls per run: X (orchestrator loop + guardrail)
- Projected: 1,000 runs/day = $X/month

## Guardrail layering
- Platform layer: Bedrock Guardrails (PII redaction, content safety)
- Application layer: brand-voice evaluator (our code)
Both apply; platform guardrails are a safety net independent of app logic.

## What I'd add for production
- Provisioned throughput if traffic is steady and latency-sensitive
- CloudWatch alarms on cost and error rate
- A VPC endpoint for Bedrock to keep traffic off the public internet
- Retry/backoff with jitter on throttling
```

Fill in your real numbers.

Commit:

```bash
git add cloud/CLOUD_DEPLOYMENT.md
git commit -m "Day 6 (wk10): SA-grade cloud deployment document"
```

---

## Day 7 (Sun): Polish + LinkedIn + Wrap

### Update the agent repo README

Add a "Cloud Deployment" section pointing to `CLOUD_DEPLOYMENT.md` and noting the agent runs on both the direct Anthropic API and AWS Bedrock. That dual-target capability is a meaningful signal: it says you understand enterprise deployment, not just prototyping.

### LinkedIn post (optional this week)

> Ported my Commerce AI agent to run on AWS Bedrock this week and wrote up the migration the way you'd document it for a customer: why Bedrock over a direct API, least-privilege IAM, regional model availability, cost projections, and guardrail layering. The model is identical to the direct API - what changes is procurement, compliance, and control. That distinction is most of what an AI Solutions Architect explains to enterprise buyers. Writeup: [link].

### Push and wrap

```bash
git add README.md
git commit -m "Day 7 (wk10): README cloud section, dual-target note"
git push
```

### Week 10 Wrap-up Checklist

- [ ] Cloud account set up, model access granted
- [ ] First cloud call working (both boto3 Converse and native-shape client for AWS)
- [ ] Agent ported to the cloud platform via client swap
- [ ] Least-privilege IAM policy applied and verified
- [ ] Cost/latency measured and projected to scale
- [ ] `CLOUD_DEPLOYMENT.md` written (SA-grade)
- [ ] You can explain why enterprises use cloud AI platforms over direct APIs
- [ ] You can explain what changes (auth/region/billing) and what doesn't (the model)

---

# WEEK 11: Certification Sprint + Polish

**Goal**: Take (or schedule) your cloud AI certification, and use any remaining time to tighten your portfolio before the Week 12-13 job-search push.

---

## Day 8 (Mon): Cert Orientation + Study Plan

### Concept: Why certs, honestly

Certs don't win offers. Skills and portfolio win offers. But certs get you *past recruiter keyword filters* and signal commitment. For a career-changer especially, "AWS Certified AI Practitioner" on your resume answers the recruiter's unspoken "is this person serious about AI?" before a human ever reads your projects. It's a filter-clearing tool, not a capability claim. Get one; don't over-invest.

### The two options (pick the one matching your Week 10 platform)

**AWS Certified AI Practitioner (AIF-C01)**:
- Foundational level, ~$100, ~30 hours of prep total
- Covers: AI/ML/GenAI fundamentals, responsible AI, Bedrock, SageMaker basics, prompt engineering, security/governance
- Exam guide: https://docs.aws.amazon.com/aws-certification/latest/examguides/ai-practitioner-01.html

**Azure AI Engineer Associate (AI-102)**:
- Associate level (harder), covers Azure AI services, Azure OpenAI, document intelligence, more
- Learning path: https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/

### Project: Schedule the exam now

Book the exam for the end of this week or early Week 12. A scheduled exam with money on the line is the forcing function that makes you actually study. For AWS AIF-C01, schedule through Pearson VUE (online proctored from home is fine).

### Study resources (AWS AIF-C01)

- Official AWS Skill Builder AIF-C01 course (free): the authoritative content
- A video course (Stephane Maarek or Frank Kane on Udemy, $15-20 on sale): efficient structured prep
- Practice exams (Tutorials Dojo or Whizlabs, $20-30): the single highest-ROI study material - they reveal the question style and your weak spots

Today: watch the first third of your video course and skim the exam guide so you know the domain weightings.

---

## Day 9-10 (Tue-Wed): Study + Practice Exams

### The efficient study loop

You already know a lot of this material from the prior 10 weeks (prompt engineering, RAG, guardrails, responsible AI, Bedrock). The cert just formalizes it and adds AWS-specific service names. So your study is mostly:

1. **Fill AWS-specific gaps**: SageMaker vs Bedrock, AWS AI services (Comprehend, Rekognition, Textract, Transcribe), AWS's responsible-AI framing, Bedrock-specific features (Knowledge Bases, Agents, Guardrails).
2. **Drill practice exams**: take one, review every wrong answer and every right-answer-you-guessed, repeat. The practice exams are the highest-leverage hours.

### Day 9: Content gaps

Work through the video course sections on AWS AI services you haven't touched. Take notes on the service-to-use-case mapping (e.g., "extract text from documents -> Textract"; "RAG with managed retrieval -> Bedrock Knowledge Bases"). The exam tests a lot of "which service for this scenario."

### Day 10: First practice exam

Take a full practice exam under timed conditions. Score it. Review every question you missed AND every one you guessed. Note your weak domains. You'll likely find you're strong on GenAI concepts (you've lived them) and weaker on AWS-specific service trivia (memorization). Spend your remaining study time on the weak domains.

---

## Day 11 (Thu): More Practice + Weak-Domain Focus

Take a second practice exam. Compare to the first - you should see improvement. Drill the domains where you're still below passing. By end of today you want to be consistently scoring above the pass threshold (around 70% for AIF-C01) on practice exams. If you're there, you're ready.

Light reading on anything still shaky. Don't cram new material the day before; consolidate what you know.

---

## Day 12 (Fri): Take the Exam

Take the exam (or, if you scheduled it for early Week 12, do a final practice exam and review today). AIF-C01 is foundational; if you've done the prior 10 weeks and drilled practice exams, you'll pass.

Pass or not, update LinkedIn: add the cert if you passed, or "studying for AWS AI Practitioner" if the exam is scheduled. Recruiters filter on both.

### If you pass

Add to your resume's certifications section and LinkedIn. A quick LinkedIn post is fine:

> Passed the AWS Certified AI Practitioner exam. The foundational concepts were familiar from building RAG systems, agents, and a cloud deployment over the last few months - the cert mostly formalized what the projects taught. On to the job search.

---

## Day 13-14 (Sat-Sun): Portfolio Polish Before the Push

With the cert done, use the weekend to tighten everything before Weeks 12-13 (the job-search push). A pre-flight check across all your repos:

### Repo audit

For each of your repos (`cert-rag-cli`, `commerce-ai-ops-agent`, `drupal-mcp-server`, `spice-voice-lora`):

- [ ] README is clear, has a one-line description, setup steps, and what it demonstrates
- [ ] Architecture diagram where relevant (Mermaid renders on GitHub)
- [ ] No secrets committed (scan for keys: `git log -p | grep -i "sk-"`)
- [ ] A demo (screenshot, GIF, or short video) where it adds clarity
- [ ] Pinned to your GitHub profile

### The "would this impress me?" test

Look at each repo as if it were someone else's that you found while hiring. Would it make you want to interview them? If a repo feels thin, either strengthen it or de-emphasize it. Three strong repos beat five mediocre ones.

### Prep the blog post queue

You have drafts from earlier weeks (retrieval strategies, MCP server) plus the content-negotiation and AGENTS.md posts from before the learning plan started. Line them up for Week 12 publishing in a sensible order. Don't publish yet; Week 12 is the coordinated push.

### Week 11 Wrap-up Checklist

- [ ] Cloud AI certification taken (or scheduled for early Week 12)
- [ ] LinkedIn updated with cert status
- [ ] All repos audited: READMEs, diagrams, no secrets, demos, pinned
- [ ] Blog post queue lined up for Week 12
- [ ] You can speak to your cloud platform's AI services at a foundational level
- [ ] Portfolio passes the "would this impress me as a hiring manager?" test

---

## A Note on Pacing for Weeks 10-11

After the conceptual intensity of the agent weeks, this phase is concrete and grounding - real infrastructure, real deployment, a clear exam target. That's a welcome change of pace. The risk here isn't difficulty; it's letting the cert study expand to fill more time than it deserves. The cert is a filter-clearing tool, not the point. Cap your study, take the exam, and move your energy to portfolio polish, because Weeks 12-13 (applications and interviews) are where the 90 days pays off.

One genuine judgment call: if your job search is urgent and you have offers or interviews materializing, the cert can wait. A scheduled-but-not-yet-taken cert still shows on LinkedIn and resumes as "in progress," which is enough to clear most filters. Don't let cert prep delay applying if real opportunities are in front of you. The portfolio and the interviews matter more than the certificate.

Weeks 12-13 are next and final: turning everything you've built into offers. You've done the hard technical work. Now you convert it. Keep moving.
