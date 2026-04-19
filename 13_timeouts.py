#!/usr/bin/env python3
"""
13 - Timeout and Connection Configuration
==========================================

Demonstrates configuring httpx timeouts and connection pools for the
OpenAI SDK. Critical for self-hosted LLM clusters where large models
can take 60-180 seconds to respond.

Blog section: "Timeouts and Connection Configuration"

Usage:
    python 13_timeouts.py
"""

import httpx
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

# Recommended production timeout configuration for MindRouter.
# MindRouter's per-attempt backend timeout is 180s, so client should
# be slightly above that to avoid premature client-side timeouts.
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=httpx.Timeout(
        200.0,       # overall timeout
        connect=5.0, # fail fast on DNS/network issues
        read=200.0,  # allow slow model generation
        write=10.0,  # request upload should be fast
        pool=10.0,   # don't wait too long for a connection slot
    ),
    max_retries=3,
)

print(f"Model: {CHAT_MODEL}")
print("Timeout config:")
print("  overall=200s, connect=5s, read=200s, write=10s, pool=10s")
print("  max_retries=3")
print("-" * 60)

response = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "What is the meaning of life? One sentence."}],
    max_tokens=64,
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens:   {response.usage.total_tokens}")
print()

# Demonstrate short timeout that would fail on slow models
print("Short timeout demo (5s — works for fast models):")
fast_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=httpx.Timeout(30.0, connect=5.0),
    max_retries=0,
)
try:
    response = fast_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=8,
    )
    print(f"  OK: {response.choices[0].message.content}")
except Exception as e:
    print(f"  Timeout: {type(e).__name__}: {e}")
