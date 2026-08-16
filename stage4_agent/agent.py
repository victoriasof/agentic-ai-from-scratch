import os
import glob
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


## ============================================================
## TOOL 1: Calculator (from Stage 2)
## ============================================================
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    return "Unknown operation"


## ============================================================
## TOOL 2: Document search (from Stage 3, now wrapped as a tool)
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


def search_docs(query):
    query_embedding = embedder.encode(query)
    scores = [cosine_similarity(query_embedding, e) for e in DOC_EMBEDDINGS]
    ranked = sorted(zip(DOC_CHUNKS, scores), key=lambda x: x[1], reverse=True)
    best_chunk, best_score = ranked[0]
    return best_chunk


## ============================================================
## Tool descriptions — this is what lets Claude choose between them
## ============================================================
tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic: add, subtract, multiply, or divide two numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                }
            },
            "required": ["a", "b", "operation"]
        }
    },
    {
        "name": "search_docs",
        "description": "Searches internal documents about the company Blorbex for relevant information. Use this for any question about Blorbex, its products, people, or history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"]
        }
    }
]

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "search_docs": search_docs,
}


## ============================================================
## The agent loop
## ============================================================
messages = []


def ask(question):
    messages.append({"role": "user", "content": question})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )

        # If Claude is done and just answering in text, stop the loop
        if response.stop_reason != "tool_use":
            answer = response.content[0].text
            messages.append({"role": "assistant", "content": answer})
            print("Claude:", answer)
            return

        # Otherwise, Claude wants to call one or more tools
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [calling {block.name} with {block.input}]")
                function_to_call = TOOL_FUNCTIONS[block.name]
                result = function_to_call(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        messages.append({"role": "user", "content": tool_results})
        # Loop back around — Claude now sees the tool result(s) and
        # decides whether it needs to call another tool, or is ready to answer


## ============================================================
## Try it
## ============================================================
ask("Who founded Blorbex, and what year?")
ask("If Blorbex had 14 employees in 2024 and hires 3 more, how many will that be?")
ask("Is that more or fewer than Blorbex had when it was founded?")

ask("How many employees did Blorbex have when it was founded, and if that number doubled, what would it be?")