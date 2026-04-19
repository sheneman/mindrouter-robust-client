#!/usr/bin/env python3
"""
04 - Embeddings
===============

Demonstrates generating vector embeddings for semantic search, RAG, and
clustering. Covers:

  - Single text embedding
  - Batch embedding of multiple texts
  - Inspecting dimensionality and token usage

Blog section: "Embeddings"

Usage:
    python 04_embeddings.py
"""

from openai import OpenAI
from config import BASE_URL, API_KEY, EMBED_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Single embedding ---
print(f"Model: {EMBED_MODEL}")
print("=" * 60)
print("Single embedding:")

response = client.embeddings.create(
    model=EMBED_MODEL,
    input="The quick brown fox jumps over the lazy dog",
)

embedding = response.data[0].embedding
print(f"  Dimensions:    {len(embedding)}")
print(f"  First 5 vals:  {embedding[:5]}")
print(f"  Tokens used:   {response.usage.total_tokens}")

# --- Batch embeddings ---
print()
print("Batch embeddings:")

texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing handles human language.",
]

response = client.embeddings.create(model=EMBED_MODEL, input=texts)

for i, item in enumerate(response.data):
    print(f"  Text {i}: {len(item.embedding)} dimensions")

print(f"  Total tokens: {response.usage.total_tokens}")
