#!/usr/bin/env python3
"""
18 - Document Reranking
========================

Demonstrates the /v1/rerank endpoint for reordering documents by
relevance to a query. Essential for RAG pipelines. This endpoint
requires vLLM backends (not available on Ollama).

Note: The OpenAI SDK doesn't have a built-in rerank method, so this
uses httpx directly.

Blog section: "Reranking and Scoring"

Usage:
    python 18_rerank.py
"""

import httpx
from config import BASE_URL, API_KEY

RERANK_MODEL = "Qwen/Qwen3-Reranker-8B"

documents = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is located in Paris, France.",
    "London is the capital of England and the United Kingdom.",
    "Moscow is the capital of Russia and sits on the Moskva River.",
]

query = "What is the capital of France?"

print(f"Model: {RERANK_MODEL}")
print(f"Query: {query}")
print("=" * 60)

response = httpx.post(
    f"{BASE_URL}/rerank",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": 3,
        "return_documents": True,
    },
    timeout=60.0,
)

if response.status_code == 200:
    data = response.json()
    print("Top 3 results:")
    for r in data["results"]:
        score = r["relevance_score"]
        doc = r.get("document") or documents[r["index"]]
        if isinstance(doc, dict):
            doc = doc.get("text", str(doc))
        print(f"  [{r['index']}] Score: {score:.4f} — {str(doc)[:70]}")
elif response.status_code == 400:
    print(f"  400 Bad Request: {response.json()}")
    print("  (Reranking requires vLLM backends, not Ollama)")
elif response.status_code == 404:
    print(f"  404: Model '{RERANK_MODEL}' not found on this cluster.")
    print("  Update RERANK_MODEL to match a reranker model on your cluster.")
else:
    print(f"  Error {response.status_code}: {response.text}")
