#!/usr/bin/env python3
"""
19 - Tokenization
==================

Demonstrates the /v1/tokenize endpoint to count input tokens before
sending a request. Useful for quota management and staying within
context window limits.

Blog section: "Tokenization"

Usage:
    python 19_tokenize.py
"""

import httpx
from config import BASE_URL, API_KEY, CHAT_MODEL

print(f"Model: {CHAT_MODEL}")
print("=" * 60)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the difference between TCP and UDP in detail."},
]

response = httpx.post(
    f"{BASE_URL}/tokenize",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": CHAT_MODEL, "messages": messages},
    timeout=30.0,
)

if response.status_code == 200:
    data = response.json()
    print(f"Input tokens:    {data['count']}")
    print(f"Model max len:   {data['max_model_len']}")
    print(f"Is estimate:     {data['is_estimate']}")
    remaining = data["max_model_len"] - data["count"]
    print(f"Remaining space: {remaining:,} tokens")
else:
    print(f"Error {response.status_code}: {response.text}")
