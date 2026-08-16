import os
import glob
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

## Small, fast, local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


## --- Step 1: Load every text file in docs/ as one chunk each ---
def load_documents():
    chunks = []
    for filepath in glob.glob("docs/*.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            chunks.append(f.read())
    return chunks


## --- Step 2: Compare two embeddings — how similar are they? ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


## --- Step 3: Find the chunk(s) most relevant to the question ---
def retrieve(question, chunks, chunk_embeddings, top_n=1):
    question_embedding = embedder.encode(question)
    scores = [cosine_similarity(question_embedding, ce) for ce in chunk_embeddings]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_n]]

def retrieve(question, chunks, chunk_embeddings, top_n=3):
    question_embedding = embedder.encode(question)
    scores = [cosine_similarity(question_embedding, ce) for ce in chunk_embeddings]

    # --- temporary debug ---
    for i, s in enumerate(scores):
        print(f"  chunk {i} score: {s:.4f}")
    # --- end debug ---

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in ranked[:top_n]]


## --- Step 4: Ask Claude, giving it only the retrieved chunk(s) as context ---
def answer_with_rag(question, chunks, chunk_embeddings):
    relevant_chunks = retrieve(question, chunks, chunk_embeddings, top_n=2)
    context = "\n\n".join(relevant_chunks)

    print(f"--- Retrieved context ---\n{context}\n")

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain the answer, say "I don't know based on the given context."

Context:
{context}

Question: {question}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


## --- Run it ---
chunks = load_documents()
chunk_embeddings = [embedder.encode(chunk) for chunk in chunks]

print(answer_with_rag("Who founded Blorbex and when?", chunks, chunk_embeddings))
print()
print(answer_with_rag("What is Blorbex's motto?", chunks, chunk_embeddings))

print(answer_with_rag("What color is Blorbex's delivery van?", chunks, chunk_embeddings))