import os
import glob
import time  # NEW
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from observability import log_interaction  # NEW
from cost_guard import check_budget  # NEW

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


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

messages = []


def ask(question):
    start = time.time()  # NEW — start the clock
    tool_calls_made = []  # NEW — track which tools get used

    messages.append({"role": "user", "content": question})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=tools,
            messages=messages
        )

        check_budget(response.usage.input_tokens, response.usage.output_tokens)  # NEW

        if response.stop_reason != "tool_use":
            answer = response.content[0].text
            messages.append({"role": "assistant", "content": answer})
            print("Claude:", answer)

            duration = time.time() - start  # NEW
            log_interaction(  # NEW
                question=question,
                tool_calls=tool_calls_made,
                answer=answer,
                duration_seconds=duration,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            return answer

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [calling {block.name} with {block.input}]")
                tool_calls_made.append(block.name)  # NEW
                function_to_call = TOOL_FUNCTIONS[block.name]
                result = function_to_call(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        messages.append({"role": "user", "content": tool_results})


ask("Who founded Blorbex, and what year?")
ask("If Blorbex had 14 employees in 2024 and hires 3 more, how many will that be?")