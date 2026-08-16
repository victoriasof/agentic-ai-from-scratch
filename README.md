# AI Agent Fundamentals

A staged learning path: each folder is one step, building from a single API call
to a full agent with tools, retrieval, memory and evaluation.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
copy .env.example .env       # Windows  (cp on macOS/Linux)
```

Then fill in your real keys in `.env`. Never commit `.env`.

## Stages

| Folder | File | What it covers |
|---|---|---|
| `stage1_first_call/` | `chat.py` | A single LLM API call |
| `stage2_tool_use/` | `tool_use.py` | Letting the model call a function |
| `stage2b_weather/` | `weather_tool.py` | A tool that hits a real external API |
| `stage3_rag/` | `rag.py`, `docs/` | Retrieval-augmented generation |
| `stage4_agent/` | `agent.py` | A loop: plan, act, observe, repeat |
| `stage5_langgraph/` | `langgraph_agent.py` | Graph-based agent orchestration |
| `stage6_vector_db/` | `vector_db.py` | Persistent embeddings storage |
| `crewai_multiagent/` | `crew.py` | Multiple agents working together |
| `stage7_authenticated_api/` | `authenticated_tool.py` | Tools behind auth |
| `stage8_evaluation/` | `evaluate.py`, `golden_dataset.py` | Measuring output quality |
| `stage9_mcp/` | `mcp_server.py` | Exposing tools over MCP |

## Notes

- One shared `venv/` and one shared `requirements.txt` across all stages.
- Run each stage from the project root so `.env` is picked up consistently.
