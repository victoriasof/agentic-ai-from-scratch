import os
import glob
import chromadb
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


## ============================================================
## Step 1: Set up Chroma — a persistent, on-disk database
## ============================================================
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection(name="blorbex_docs")


## ============================================================
## Step 2: Load and add documents — only if the collection is empty
## (so you don't re-add the same docs every time you run this)
## ============================================================
def index_documents():
    if collection.count() > 0:
        print(f"Collection already has {collection.count()} documents — skipping re-index.")
        return

    filepaths = glob.glob("docs/*.txt")
    documents = []
    ids = []
    embeddings = []

    for i, filepath in enumerate(filepaths):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(text)
        ids.append(f"doc_{i}")
        embeddings.append(embedder.encode(text).tolist())

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Indexed {len(documents)} documents.")


## ============================================================
## Step 3: Search — this replaces your entire hand-built
## cosine_similarity + sorted() logic from Stage 3/4
## ============================================================
def search_docs(query, n_results=1):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0][0]  # best match, as plain text


## ============================================================
## Step 4: Same RAG answer function as Stage 3, unchanged
## ============================================================
def answer_with_rag(question):
    context = search_docs(question)
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


## ============================================================
## Run it
## ============================================================
index_documents()
print(answer_with_rag("Who founded Blorbex and when?"))

print(collection.count())