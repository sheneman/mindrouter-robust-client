#!/usr/bin/env python3
"""
02 - Streaming Chat Completion
==============================

Demonstrates streaming (Server-Sent Events) so tokens appear as they are
generated, rather than waiting for the full response. Covers:

  - Synchronous streaming with ``stream=True``
  - Accumulating the full response for logging
  - Detecting finish_reason in the final chunk

Blog section: "Streaming Responses"

Usage:
    python 02_streaming.py
"""

from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

print(f"Model: {CHAT_MODEL}")
print("-" * 60)

stream = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "Write a short poem about the stars."}],
    stream=True,
    max_tokens=512,
)

full_response = ""
finish_reason = None

for chunk in stream:
    choice = chunk.choices[0]
    delta = choice.delta.content
    if delta is not None:
        print(delta, end="", flush=True)
        full_response += delta
    if choice.finish_reason:
        finish_reason = choice.finish_reason

print()
print("-" * 60)
print(f"Finish reason: {finish_reason}")
print(f"Total chars:   {len(full_response)}")
