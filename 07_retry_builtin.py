#!/usr/bin/env python3
"""
07 - SDK Built-in Retries
=========================

Demonstrates the simplest retry strategy: the OpenAI SDK's built-in
``max_retries`` parameter. The SDK automatically retries on 429, 5xx,
timeouts, and connection errors with exponential backoff and jitter.

Blog section: "Retry Strategies" (Level 1)

Usage:
    python 07_retry_builtin.py
"""

from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

# Increase retry count from the default of 2
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    max_retries=5,
)

print(f"Model: {CHAT_MODEL}")
print(f"Max retries: 5 (SDK handles backoff + jitter automatically)")
print("-" * 60)

# Normal request — SDK will retry transparently on transient errors
response = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
    max_tokens=16,
)
print(f"Response: {response.choices[0].message.content}")

# Per-request override
print()
print("Per-request override (max_retries=8):")
response = client.with_options(max_retries=8).chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "What is 3 + 3? Answer with just the number."}],
    max_tokens=16,
)
print(f"Response: {response.choices[0].message.content}")
