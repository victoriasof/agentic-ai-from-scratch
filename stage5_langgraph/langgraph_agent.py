import os
import glob
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


## ============================================================
## Same document loading as Stage 4 — unchanged
## ============================================================
def load_documents():
    chunks = []
    for filepath in glob.glob("docs/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks.append(f.read())
    return chunks


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


DOC_CHUNKS = load_documents()
DOC_EMBEDDINGS = [embedder.encode(chunk) for chunk in DOC_CHUNKS]


## ============================================================
## Tools — this is the first real difference from Stage 4.
## The @tool decorator replaces your manual "tools = [...]" schema
## dictionary. LangGraph reads the function's docstring and type
## hints to build that same schema automatically.
## ============================================================
@tool
def calculator(a: float, b: float, operation: str) -> str:
    """Performs basic arithmetic: add, subtract, multiply, or divide two numbers.

    Args:
        a: The first number
        b: The second number
        operation: One of "add", "subtract", "multiply", "divide"
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        return str(a / b)
    return "Unknown operation"


@tool
def search_docs(query: str) -> str:
    """Searches internal documents about the company Blorbex.
    Use this for any question about Blorbex, its products, people, or history.

    Args:
        query: What to search for
    """
    query_embedding = embedder.encode(query)
    scores = [cosine_similarity(query_embedding, e) for e in DOC_EMBEDDINGS]
    ranked = sorted(zip(DOC_CHUNKS, scores), key=lambda x: x[1], reverse=True)
    return ranked[0][0]


## ============================================================
## The agent itself — this ONE function call replaces your
## entire Stage 4 "while True" loop, the tool_use_id matching,
## the messages.append bookkeeping, all of it.
## ============================================================
model = ChatAnthropic(model="claude-sonnet-4-6")
agent = create_react_agent(model, tools=[calculator, search_docs])


## ============================================================
## Run it — LangGraph manages the conversation history internally
## ============================================================
def ask(question, history):
    history.append({"role": "user", "content": question})
    result = agent.invoke({"messages": history})
    # result["messages"] contains the full updated conversation
    final_message = result["messages"][-1]
    print("Claude:", final_message.content)
    return result["messages"]


history = []
history = ask("Who founded Blorbex, and what year?", history)
history = ask("If Blorbex had 14 employees in 2024 and hires 3 more, how many will that be?", history)

print(agent.get_graph().draw_ascii())