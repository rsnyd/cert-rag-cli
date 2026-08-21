# Weeks 12-13 Detailed Walkthrough: Portfolio & Apply (The Conversion)

**For Rob Snyder. Working tutorial, not just a plan. Aim: by end of Week 13 you've published your blog posts, finalized your resume and LinkedIn around the new portfolio, submitted 20+ applications, run 2+ mock interviews, and have first-round interviews in motion.**

**Why this phase matters**: You've done the hard technical work. Weeks 1-11 built the capability and the evidence. These two weeks convert that into a job search. This is the phase career-changers most often under-invest in - they keep building instead of applying. Resist that. The portfolio is good enough. Publishing and applying is now the highest-leverage work, and it's a different skill than coding: it's narrative, positioning, and persistence.

This is two weeks (14 days). Week 12 is publish + finalize materials. Week 13 is apply + interview prep + outreach.

---

## Before You Start: Inventory Check (15 minutes)

Coming into Week 12 you should have, from Weeks 1-11:

- [ ] `cert-rag-cli` - SQF compliance RAG with three retrieval strategies, clause-citation + refusal evals, Langfuse, three implementations
- [ ] `commerce-ai-ops-agent` - multi-tool agent, raw + LangGraph + CrewAI, guardrails, cloud deployment
- [ ] `drupal-mcp-server` - published MCP server
- [ ] `spice-voice-lora` - published fine-tune
- [ ] A cloud AI cert (taken or scheduled)
- [ ] Blog post drafts: retrieval strategies, MCP server (from the learning weeks)
- [ ] Earlier drafts: content negotiation, AGENTS.md governance (from before the learning plan)
- [ ] The AI SA resume draft and a list of target companies (from the original pivot work)

That's a strong inventory. The job now is to make it legible and get it in front of people.

---

# WEEK 12: Publish + Finalize Materials

**Goal**: Publish your blog posts in a deliberate sequence, finalize your resume and LinkedIn around the portfolio, and prepare your application materials. By Sunday you're ready to apply at volume.

---

## Day 1 (Mon): The Publishing Plan + First Post

### Concept: Sequencing beats dumping

You have four to five publishable posts. Publishing them all at once wastes them - they compete for the same audience attention and your LinkedIn feed shows one burst then silence. Publishing one every 2-3 days over two weeks creates a sustained "this person is active and producing" signal that hiring managers notice when they look you up. Each post is also a fresh reason to appear in your network's feed.

### The publishing order (most-to-least differentiated)

1. **MCP server post** - "I Built an MCP Server for Drupal Development." Lead with this. It's your most differentiated artifact and MCP is hot. Best chance of organic reach.
2. **AGENTS.md governance post** - "A Governance Framework for AI-Assisted Enterprise Development." Broad appeal, viral potential, positions you as a methodology thinker.
3. **Retrieval strategies post** - "Retrieval Strategies for a Compliance RAG: Citing the Right Clause." Demonstrates rigor and evaluation discipline.
4. **Content negotiation post** - "Content Negotiation for AI Crawlers." Forward-looking, shows you think ahead.

Space them ~3 days apart: Mon, Thu, Sun, then Wed of Week 13.

### Concept: Where to publish

Cross-post each to maximize reach:

- **Your blog** (dev.to or Hashnode) - the canonical home, owns the SEO
- **LinkedIn** - either the full post or a strong summary linking to the blog. This is where hiring managers see it.
- **Hacker News** (Show HN for the MCP server, regular submission for others) - high variance, but a front-page hit is enormous reach. Best submission time is weekday mornings US Eastern.
- **r/drupal** for the Drupal-flavored posts; relevant subreddits for others
- **Relevant Discord/Slack communities** (MCP, LangChain, AI engineering) where sharing is welcome

### Project: Polish and publish post #1 (MCP server)

Take your `BLOG_POST_DRAFT.md` from the MCP repo. Polish it:

- Strong title and first paragraph (the hook - most readers decide in two sentences)
- Clear structure with subheads
- Code snippets that are correct and minimal
- The "payoff" screenshots (the AI assistant using your tools)
- A clear call to action at the end (repo link, invitation for feedback)
- Scan for em dashes; replace with hyphens
- A short author bio linking your other work

Publish to your blog. Then write the LinkedIn version (a 150-word summary + link, or the full post natively). Post it. Submit to Hacker News as "Show HN: An MCP server for Drupal development." Cross-post to r/drupal.

### Engagement matters

After posting, respond to every comment thoughtfully for the first 24-48 hours. Early engagement drives algorithmic reach on LinkedIn especially. This is part of the work, not optional.

Track where you posted and any responses in a simple file so you can follow up.

---

## Day 2 (Tue): Finalize the Resume

### Concept: The resume the portfolio earned

You drafted an AI SA resume early on, before the portfolio existed. Now you have real artifacts to put in it. The "Selected AI Projects" section that was aspirational is now concrete.

### Project: Update the resume

Open your `RobertSnyder_AI_SA_Resume.docx` and make these updates:

**Selected AI Projects section** - now populated with real, linked work:

- *Commerce AI Ops Agent* - Multi-tool agent (orchestrator-workers) for e-commerce merchandising, built in raw Anthropic SDK and LangGraph, with brand-voice guardrails, Langfuse observability, and outcome evaluation. Deployed on AWS Bedrock. [github link]
- *Drupal MCP Server* - Model Context Protocol server exposing Drupal development tools (hook explanation, service parsing, entity scaffolding) to any MCP-compatible AI assistant. [github link]
- *SQF Compliance RAG + Evaluation Framework* - Retrieval system over SQF food-safety certification documents (PDF/DOCX, clause-aware chunking) comparing three strategies (vanilla, hybrid, rerank), with a 34-question golden set plus 5 refusal probes, deterministic clause-citation and refusal metrics, and five-axis LLM-as-judge scoring. [github link]
- *Brand-Voice LoRA* - QLoRA fine-tune adapting a small model to a specific brand voice, with an honest fine-tune-vs-prompting evaluation. [HF link]

**Applied AI Skills section** - now everything is demonstrated, not claimed. You can drop any hedging.

**Certifications** - add your cloud AI cert (or "in progress").

**Summary** - tighten to reflect what you can now prove: "Shipped production RAG systems, multi-tool agents, an MCP server, and cloud deployments."

Fill in the real GitHub/HuggingFace URLs (replace the `[your-handle]` placeholders). Export a clean PDF for applications and keep the docx for edits.

### Tailoring note

Keep one master resume, but plan to tweak the summary and project emphasis per application cluster: lead with commerce-AI projects for commerce companies, the MCP server for AI infra companies, the governance/agent work for consultancies.

Save both formats. You'll reference them all of Week 13.

---

## Day 3 (Wed): Finalize LinkedIn

### Concept: LinkedIn is where hiring managers verify you

After a recruiter screens your resume, the hiring manager Googles you. LinkedIn is usually the first result. It needs to tell the same story as your resume, reinforced with the posts you're publishing this week.

### Project: Overhaul the LinkedIn profile

- **Headline**: `AI Solutions Engineer | Applied AI for E-commerce & Content Systems | RAG, Agents, MCP` (or similar - lead with the role you want, not the role you have)
- **About**: rewrite to the pivot narrative. 20+ years building production systems, now focused on applied AI. Name the concrete artifacts. Make it scannable - short paragraphs, no wall of text.
- **Featured section**: pin your three strongest items - the MCP server repo, the agent repo, and your best blog post. This is the first thing visitors see; make it your best work.
- **Experience**: update your current role to reflect the AI work (the spicesinc_markdown content negotiation layer, the AGENTS.md governance, the AI-assisted development - all real and all on your current job).
- **Skills**: add the AI skills so you match recruiter searches (RAG, LangChain, LangGraph, MCP, LLM, prompt engineering, AWS Bedrock, vector databases).
- **Open to work**: turn on the "open to work" signal (recruiters-only visibility if you don't want it public).

### The consistency check

Read your resume and LinkedIn side by side. Same story? Same projects? Same positioning? Inconsistency reads as carelessness. Align them.

---

## Day 4 (Thu): Publish Post #2 + Build the Application Tracker

### Project: Publish post #2 (AGENTS.md governance)

Polish and publish your AGENTS.md governance post following the same process as Day 1. Before publishing, make sure the sanitized AGENTS.md template is actually on GitHub and linked from the post - the post drives traffic to the template, and the template is what a hiring manager evaluates. This post has the broadest potential appeal; give it your best title and opening.

Cross-post, engage with comments, track responses.

### Project: Build an application tracker

Before applying at volume, set up a simple tracker (a spreadsheet or a Notion table). Columns:

- Company
- Role title
- Tier (1/2/3 from your target list)
- Date applied
- Source (where you found it / how you applied)
- Referral? (warm intro path if any)
- Resume version used
- Status (applied / screen / interview / offer / rejected)
- Next action + date
- Notes

A tracker keeps you from applying twice, helps you follow up, and surfaces patterns (which sources convert). Career changers who track outperform those who spray and pray.

### Pre-load the target list

From your original pivot work, you have a tiered target list. Load it into the tracker now so Week 13 is pure execution:

- **Tier 2/3 (apply first)**: commerce AI vendors (Algolia, Bloomreach, Klevu, Constructor.io, Klaviyo, Attentive), AI infra (Pinecone, Weaviate, Langfuse, Vellum, Humanloop)
- **Consultancies**: Slalom, Lullabot, Credera, EPAM, Thoughtworks
- **Tier 1 (apply once you have momentum)**: AWS Bedrock SA, Azure AI CSA, Anthropic FDE, OpenAI SE, Glean, Writer, Moveworks

---

## Day 5 (Fri): Application Materials Kit

### Concept: Reduce per-application friction

If each application takes an hour, you'll apply to five companies and burn out. Build reusable materials so each application takes 15 minutes: tailor, don't rewrite.

### Project: Build the kit

1. **Master resume PDF** (done Day 2) + 2-3 cluster variants (commerce, infra, consultancy emphasis)
2. **Cover letter template** with swappable paragraphs. A short, strong template:
   - Para 1: the pivot in one sentence + why this company specifically
   - Para 2: the most relevant 2-3 portfolio artifacts with links
   - Para 3: the unfair advantage (20 years of production systems + real e-commerce ownership + recent shipped AI work)
   - Keep it under 200 words. Most are skimmed.
3. **A "portfolio one-pager"** - a single page (or a clean GitHub profile README) linking all four projects with one-line descriptions. Drop this link into applications that allow it.
4. **Outreach message templates** for warm intros and cold-but-targeted messages to hiring managers (see Day 9).
5. **A short answer bank** for common application questions ("why are you interested in this role," "describe a relevant project") so you're not writing from scratch each time.

### The GitHub profile README

If you haven't, create a profile README (a repo named exactly your username with a README.md). Make it a clean landing page: who you are, the pivot, the four pinned projects with descriptions and links, your blog, your cert. This is often the first thing a technical hiring manager clicks. Make it good.

---

## Day 6 (Sat): Mock Interview #1 + Whiteboard Practice

### Concept: Interviewing is a separate skill

You can build all of this and still stumble in interviews if you don't practice articulating it. SA/SE interviews have predictable shapes. Practice them out loud, ideally with another person.

### The common SA/SE interview formats

1. **Portfolio walkthrough**: "Tell me about a project you're proud of." Have a crisp 3-minute story for each major artifact: the problem, your approach, the tradeoffs, the result. Practice these until they're smooth.
2. **System design**: "Design an AI-powered customer service system for a mid-market e-commerce company." This is the most common SA whiteboard. You've literally built adjacent things; structure your answer as: clarify requirements -> propose RAG + agent architecture -> address retrieval, evaluation, guardrails, cost, observability, human handoff. You can run this cold because you've done the pieces.
3. **Conceptual depth**: "When would you fine-tune vs RAG?" "How would you evaluate an LLM feature before shipping?" "How would you reduce the cost of a high-traffic LLM app by 50%?" You have real answers from your projects.
4. **Customer-facing roleplay**: "Explain RAG to a non-technical executive in three minutes." SE roles especially test this. Practice the plain-language version.

### Project: Run mock interview #1

Use Pramp (https://www.pramp.com) for a peer mock, or recruit a technical friend. Focus on the system-design prompt and two portfolio walkthroughs. Record yourself if solo. Watch for: rambling, jargon without explanation, failing to state tradeoffs, not asking clarifying questions.

### Whiteboard the customer-service design twice

On paper, design the AI customer service system end to end. Do it once today, once tomorrow. By the second pass it should flow. Key elements to always hit: intent classification/routing, RAG over knowledge base, tool use (order lookup, refund), evaluation strategy, guardrails (safety + brand), human-in-the-loop escalation, observability, cost management.

---

## Day 7 (Sun): Publish Post #3 + Week 12 Wrap

### Project: Publish post #3 (retrieval strategies)

Polish and publish "Retrieval Strategies for a Compliance RAG: Citing the Right Clause." This one demonstrates evaluation rigor - the skill that separates serious AI engineers from dabblers. Lead with what is distinctive: clause-level ground truth, a judge-free clause-hit metric, the refusal probe set, and a comparison led by citation accuracy rather than answer quality. Make sure the results tables are clean and the "what this didn't measure" honesty is intact (it signals senior judgment).

Cross-post, engage.

### Week 12 Wrap-up Checklist

- [ ] Three blog posts published (MCP, AGENTS.md, retrieval), each cross-posted and engaged
- [ ] Resume finalized with real, linked portfolio projects + cert
- [ ] LinkedIn overhauled: headline, about, featured, skills, open-to-work
- [ ] Resume and LinkedIn tell a consistent story
- [ ] Application tracker built and pre-loaded with target list
- [ ] Application materials kit ready (resume variants, cover template, one-pager, outreach templates)
- [ ] GitHub profile README as a clean landing page
- [ ] One mock interview done
- [ ] Customer-service system design whiteboarded twice
- [ ] You have a crisp 3-minute story for each major project

---

# WEEK 13: Apply + Interview + Outreach

**Goal**: Apply at volume, run more interviews, activate your network, and get first-round interviews in motion. This is execution week.

---

## Day 8 (Mon): First Application Batch (15 applications)

### Concept: Volume with targeting

Apply broadly within your tier 2/3 list, but tailor lightly each time (15 minutes each, using your kit). The goal today is 15 quality applications, not 50 sloppy ones. Lead with the companies where your commerce + AI background is the clearest fit - the commerce AI vendors and AI infra companies.

### Project: Apply to 15

Work through your tracker's tier 2/3 list. For each:

1. Read the actual job description; note 2-3 specific requirements
2. Pick the matching resume variant
3. Tailor the cover letter's company-specific sentence and the 2-3 highlighted projects
4. Apply; log it in the tracker with date and source
5. Check for a warm-intro path (Day 9) before applying cold to your top-choice companies

### Where to find roles

- Company career pages directly (best signal, least competition)
- AI Jobs board (https://aijobs.net)
- Wellfound (https://wellfound.com) for startups
- LinkedIn Jobs (filter for "AI Solutions Engineer," "Forward Deployed Engineer," "Applied AI Engineer," "AI Implementation Engineer")

### Don't wait for responses

Applications have long latency. Apply, log, move on. You'll start hearing back in 1-3 weeks. The work this week is volume and outreach, not refreshing your inbox.

---

## Day 9 (Tue): Warm Outreach + Network Activation

### Concept: Referrals convert far better than cold applications

A referred candidate is many times more likely to get an interview than a cold applicant. You have a network from 20+ years in the industry and the Drupal community. Activate it.

### Project: Map and activate your network

1. **List warm connections** at or near your target companies. Specifically:
   - Anyone you know at Lullabot (Drupal shop with an AI practice - a natural bridge)
   - Drupal/Acquia community contacts now at AI companies
   - Former colleagues, clients, anyone in the e-commerce or AI space
2. **Reach out personally** (not a mass message). For each, a short, specific note:
   > "Hi [name] - I've spent the last few months pivoting hard into applied AI engineering: shipped a multi-tool agent, an MCP server, and a few RAG systems [link]. I noticed [their company] is hiring for [role]. Would you be open to a quick chat about the team, or pointing me to the right person? Either way, good to reconnect."
3. **Engage with target companies' content** on LinkedIn before applying - thoughtful comments on their posts get you noticed by their team.

### The Lullabot angle specifically

Lullabot is worth a dedicated effort: they're a respected Drupal shop that has built an AI practice, which makes you an unusually strong fit (deep Drupal + new AI skills). If you have any connection there, use it. If not, a thoughtful cold outreach to someone on their AI team, leading with your Drupal MCP server, is a strong play.

### Outreach to hiring managers directly

For your top 5 target roles, find the hiring manager or team lead on LinkedIn and send a short, specific message leading with the most relevant artifact. This bypasses the resume black hole. Keep it under 100 words, lead with value (a relevant project), and make a small specific ask (a 15-minute chat).

---

## Day 10 (Wed): Publish Post #4 + Second Application Batch

### Project: Publish post #4 (content negotiation)

Publish "Content Negotiation for AI Crawlers" - your most forward-looking post. It signals you think ahead of the curve, which is exactly the SA mindset. Cross-post and engage.

### Project: Second batch (10 more applications)

Another 10 applications from your tracker. By now you may expand slightly toward tier 1 if you've gotten any positive signal from the first batch. Keep tailoring, keep logging.

You're now at 25 applications submitted. That's a healthy volume for a focused two-week push.

---

## Day 11 (Thu): Mock Interview #2 + Deepen Prep

### Project: Mock interview #2

A second mock, focused on whatever felt weakest in the first. If system design was rough, drill that. If the portfolio walkthroughs rambled, tighten them. If you froze on the "explain RAG to an executive" prompt, practice the plain-language version until it's smooth.

### Deepen on the likely-hardest questions

Prepare strong answers, out loud, for:

- "Walk me through how you'd evaluate whether an LLM feature is ready to ship." (You have a real answer: golden sets, LLM-as-judge, the axes, what it doesn't catch.)
- "A customer's RAG system gives wrong answers. How do you debug it?" (Retrieval vs generation failure; you've done this; trace it, check retrieved chunks, check the eval.)
- "When is an agent the wrong choice?" (When a simpler pattern works; agents cost more and are less predictable; you measured this.)
- "How would you reduce this LLM app's cost by half?" (Prompt caching, model routing to cheaper models for easy queries, retrieval to shrink context, batching.)
- "Tell me about a time you shipped something to production." (The Drupal 11 migration, spicesinc_markdown, the AGENTS.md governance - all real.)

### Prepare your questions for them

Interviews are two-way. Have sharp questions ready: "How does the team measure whether an AI feature is working?" "What's the hardest customer-facing AI problem you're working on?" "How do you balance shipping speed against evaluation rigor?" Good questions signal seniority.

---

## Day 12 (Fri): Follow-ups + Pipeline Management

### Concept: The fortune is in the follow-up

Many applications stall not from rejection but from inertia. A polite follow-up moves things.

### Project: Work the pipeline

- **Follow up** on applications from 1+ weeks ago with no response (if you have a contact). A short note reaffirming interest.
- **Respond fast** to any recruiter or hiring-manager replies. Speed signals enthusiasm and keeps you top of pipeline.
- **Schedule** any interviews that have come in. Block prep time before each.
- **Update the tracker** - move statuses, set next-action dates. The tracker is your command center now.

### Triage your energy

If interviews are materializing, shift energy from new applications to interview prep - a scheduled interview is worth more than ten new applications. If nothing's moving yet (normal at two weeks; latency is real), keep applying and keep doing outreach, which converts faster than cold applications.

---

## Day 13 (Sat): Reflection + Adjust

### Concept: Two weeks of data tells you something

By now you have signal: response rates, which sources convert, which roles reply. Use it.

### Project: Review and adjust

- **What's converting?** If commerce AI vendors are responding and infra companies aren't (or vice versa), lean into what's working.
- **What's the resume doing?** If you're getting no responses at all across 25 applications, the resume or positioning may need work - have a trusted contact review it. (More likely at two weeks: it's just latency. Don't over-correct on thin data.)
- **Is the outreach working?** Warm intros and direct hiring-manager messages should be converting better than cold applications. If you haven't done much outreach, that's the gap to close.
- **Adjust the target mix** for the coming weeks based on what's responding.

### The honest checkpoint

The 90-day plan ends here, but the job search continues past it. The realistic outcome of these 13 weeks is not "signed offer on day 91" - it's "strong portfolio, materials, and an active pipeline with interviews in motion." Landing the role often takes another 4-8 weeks of interviewing past the plan. That's normal and not a failure. You've built the thing that gets you in the door; now you work the funnel.

---

## Day 14 (Sun): Consolidate + The Path Forward

### Project: Set up the ongoing rhythm

The intense build phase is done. Shift to a sustainable job-search cadence you can maintain for as long as it takes:

- **Weekly**: 10-15 new applications, 3-5 outreach messages, one piece of engagement (a comment, a small post, an answer in a community)
- **Per interview**: dedicated prep block, post-interview notes in the tracker, a thank-you note within 24 hours
- **Monthly**: one new small portfolio piece or blog post to keep the "active and producing" signal alive and give your network a reason to see you

### Keep building, but lightly

You don't need another big project to get hired - you have enough. But shipping one small thing a month (a new MCP tool, a blog post on something you learned in an interview, an experiment with a new model) keeps your momentum and your profile fresh. It also gives you fresh material for interviews.

### The strategic reminder

Your unfair advantages haven't changed, and you should lead with them in every conversation: 20+ years shipping production systems, real e-commerce ownership (you ARE the customer that commerce-AI vendors serve), an instructional-design background that makes you a natural at the teaching half of SE work, and now a concrete AI portfolio that most career-changers can't match. The combination is rare. Most candidates have the AI skills OR the production experience OR the domain depth. You have all three. Position accordingly.

### Weeks 12-13 Wrap-up Checklist

- [ ] All four blog posts published, cross-posted, engaged
- [ ] Resume and LinkedIn finalized and consistent
- [ ] 25+ applications submitted and tracked
- [ ] Warm network activated; direct outreach to 5+ hiring managers
- [ ] Lullabot and similar bridge-companies specifically pursued
- [ ] Two mock interviews done; strong answers prepared for common questions
- [ ] System-design and "explain to an executive" prompts practiced
- [ ] Pipeline tracker active with next-actions
- [ ] A sustainable ongoing job-search rhythm established
- [ ] First-round interviews in motion (the realistic Day-91 outcome)

---

## A Note on Pacing for Weeks 12-13 - and Beyond

These two weeks ask for a different mode than the prior eleven. Building is solitary and concrete; job-searching is social, ambiguous, and rejection-heavy. That's uncomfortable for a lot of engineers, and the temptation is to retreat to building "just one more project." Resist it. You have enough. The work now is narrative and persistence.

A few honest truths to hold:

- **Rejection is volume, not verdict.** Most applications go nowhere for reasons unrelated to you. The funnel is wide at the top for everyone. Keep filling it.
- **Outreach beats applications.** If you do one thing well in these two weeks, make it the warm-intro and direct-hiring-manager outreach. It converts several times better than cold applications.
- **The plan ends; the search doesn't.** Day 91 is not a deadline for an offer. It's the day you have a complete portfolio, polished materials, and an active pipeline. Landing takes as long as it takes - often another month or two of interviewing. That's the normal shape of a career pivot, not a failure of the plan.

You started this 13 weeks ago as an experienced engineer with a credibility gap in AI. You're ending it with four real AI artifacts, a cloud deployment, a cert, four published posts, and the vocabulary to discuss agents, RAG, evaluation, MCP, and production AI from genuine experience. That gap is closed. Now go convert it.

You've done the hard part. Keep moving - and good luck.
