# Agentic AI From Scratch

Nine stages, building from a single API call up to a working agent with tools, memory, retrieval, multi-agent orchestration, a real authenticated API, evaluation, and MCP. Everything is built in plain Python first. Frameworks (LangGraph, CrewAI, Chroma) only get introduced once the thing they automate already exists and is understood.

## What this shows

- How tool-calling actually works, because I built the request → decide → execute → respond loop myself before using any framework for it
- A RAG pipeline built from scratch, including a real retrieval bug I found and fixed (details below), not just a version that happened to work
- Enough understanding to judge whether a framework is actually pulling its weight at a given point, instead of assuming it's better by default
- A multi-agent crew where one agent (the Reviewer) initially couldn't work properly, and I fixed the cause 
- Basic things a real system needs: a small evaluation set, logging, and a cost guard that I tested

## Setup

```cmd
git clone https://github.com/victoriasof/agentic-ai-from-scratch.git
cd agentic-ai-from-scratch

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
copy .env.example .env
```

Fill in your real keys in `.env`:
- `ANTHROPIC_API_KEY` from console.anthropic.com (you'll need to top up credits — the free trial credit runs out fast)
- `GITHUB_TOKEN` from github.com/settings/tokens, no scopes needed, only used for Stage 7's public repo lookups

Never commit `.env`.

**Windows notes**, since that's what this was built on:
- Use `python`, not `python3` — there's no `python3` alias on Windows
- Activate the venv with `venv\Scripts\activate`, not the Mac/Linux `source venv/bin/activate`
- If PowerShell refuses to activate the venv, either switch to Command Prompt or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once

Each stage folder needs its own copy of `docs/` (three short `.txt` files about a made-up company, Blorbex). They're small, so it made more sense to duplicate them per stage than to fight with relative paths across folders. Run a stage from inside its own folder:

```cmd
cd stage4_agent
python agent.py
```

`.env` at the project root gets found automatically no matter which folder you run from — `python-dotenv` looks in the current folder and walks upward until it finds one.

One shared `venv/` and one shared `requirements.txt` cover everything except Stage 6b — see below for why that one's different.

## The stages

| Folder | File(s) | What it covers |
|---|---|---|
| `stage1_first_call/` | `chat.py` | A single API call, no tools |
| `stage2_tool_use/` | `tool_use.py` | First tool call — a local calculator function |
| `stage2b_weather/` | `weather_tool.py` | A real external API call, plus the first working memory |
| `stage3_rag/` | `rag.py`, `docs/` | RAG from scratch — chunking, embeddings, cosine similarity, no vector database |
| `stage4_agent/` | `agent.py` | Full agent loop — multiple tools, memory, honest "I don't know" when the data isn't there |
| `stage5_langgraph/` | `langgraph_agent.py` | Same Stage 4 agent, rebuilt in LangGraph, with a graph diagram printed out |
| `stage6_vector_db/` | `vector_db.py` | Swaps the hand-built search from Stage 3 for a real vector database (Chroma) |
| `stage6b_crewai_multiagent/` | `crew.py` | Three agents — Planner, Writer, Reviewer — with a real fix applied along the way |
| `stage7_authenticated_api/` | `authenticated_tool.py` | A real API that needs a token (GitHub), with proper error handling |
| `stage8_evaluation/` | `evaluate.py`, `golden_dataset.py`, `observability.py`, `cost_guard.py` | A small evaluation set, logging, and a cost cutoff I actually tested |
| `stage9_mcp/` | `mcp_server.py`, `mcp_client.py` | The calculator tool exposed as an MCP server, called by a separate client |

## Things that broke and what I did about them

Worth writing down honestly, since none of this worked first try and the fixes taught me more than a clean run would have.

**RAG picked the wrong chunk (Stage 3).** With three short documents that all mention the same company, the small embedding model (`all-MiniLM-L6-v2`) scored the right answer for a motto question just barely *below* the wrong one — 0.5553 vs 0.5630. I only found this by printing the actual similarity scores instead of guessing why the answer was wrong. Fixed it by pulling the top 2 chunks instead of just the top 1, so a near-miss like that doesn't lose the right answer entirely. This isn't a bug in my code — it's a real limit of small embedding models on short, similar text, and it's exactly why bigger systems use bigger embedding models.

**CrewAI kept asking for an OpenAI key (Stage 6b).** Even though I never touched OpenAI, CrewAI assumes any plain model name is an OpenAI model unless told otherwise. Fixed by writing the model as `"anthropic/claude-sonnet-4-6"` instead of just `"claude-sonnet-4-6"`.

**Then it still failed, because Anthropic support isn't installed by default (Stage 6b).** CrewAI treats it as an optional extra. Fixed with `pip install "crewai[anthropic]"`.

**Python 3.14 couldn't install CrewAI at all (Stage 6b).** `numpy`, one of CrewAI's dependencies, had no ready-made install for Python 3.14 yet, and there's no C compiler on a normal Windows machine to build it from source. Fixed by making a second, separate environment just for this stage, using Python 3.10 instead:
```cmd
py -3.10 -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
pip install "crewai[anthropic]"
```
Every other stage uses the normal `venv` (3.14). Only `stage6b_crewai_multiagent/` needs `venv310`.

**The Reviewer agent couldn't actually review anything (Stage 6b).** In the first version, the Reviewer only ever saw the Writer's finished paragraph as plain text — no access to the real documents, no tool to check anything with. So it did the honest thing and refused to approve it, saying it had no way to verify the facts. That's not a broken agent, that's a badly wired one. Fixed by giving the Reviewer the same document-search tool the Writer had, so it could actually check things itself instead of taking the Writer's word for it.

**One evaluation test "failed" even though the agent was right (Stage 8).** I asked a question with no answer in the documents, expecting the agent to say so — and it did, correctly. But my test was checking for exact phrases like "don't know" or "no information," and Claude said "the internal documents don't appear to contain..." instead — same meaning, different words, so the test called it a fail. I left this one as-is rather than fixing the wording, because it's a more honest demonstration of a real limitation: simple string-matching can't tell when two answers mean the same thing, only when they're written the same way.

**The MCP SDK changed its API right as I was building this stage (Stage 9).** A new major version (2.0.0) renamed `FastMCP` and moved where it lives, which broke the original code with an error that didn't obviously point to the real cause. Fixed by pinning an older version:
```cmd
pip install "mcp<2.0.0"
```

## What's not here on purpose

This is a learning project, not something meant to run in production. Missing on purpose:
- No token-refresh handling for OAuth2 — Stage 7 uses a simpler, permanent token instead, with notes on what OAuth2 would add
- No retry logic for failed API calls
- No actual deployment, everything runs locally
- The evaluation set is four questions, not a real test suite
- `requirements.txt` reflects the main `venv` only — `venv310` for Stage 6b has its own separate installs

## What I'd do next

- Make the Reviewer in Stage 6b stricter (a more skeptical backstory) and see how much that alone changes its output, separate from what tools it has
- Give the Stage 9 MCP server more than one tool, so `list_tools()` actually has something to discover beyond a single function
- Add a proper LLM-as-judge check to Stage 8, since plain string-matching clearly isn't enough
- Try wrapping the Stage 7 pattern around something that actually needs OAuth2, now that the basic tool-calling shape is second nature

## Why I built it this way

Building each piece by hand before touching a framework meant that when the frameworks did show up, I could actually tell what they were doing instead of just trusting them. The LangGraph version in Stage 5 was recognizably my own loop from Stage 4, just drawn differently. The CrewAI bug in Stage 6b took minutes to figure out because the tool-calling mechanic underneath it was already familiar — I wasn't learning the concept and debugging it at the same time.

Nothing here touches real data or runs unsupervised. Every tool works on a small, made-up dataset, and every decision an "agent" makes — which tool to use, how many times, whether to answer at all — is something Claude actually decided while running, not something I hardcoded in advance. That's the actual meaning of "agentic".
