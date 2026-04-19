#!/usr/bin/env python3
"""
16 - Thinking / Reasoning Mode
================================

Demonstrates enabling thinking/reasoning mode on models that support it
(Qwen3, Qwen3.5, GPT-OSS). The model's chain-of-thought reasoning appears
in a separate ``reasoning_content`` field.

Blog section: "Thinking and Reasoning Mode"

Usage:
    python 16_thinking_mode.py
"""

from openai import OpenAI
from config import BASE_URL, API_KEY, LARGE_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

print(f"Model: {LARGE_MODEL}")
print("=" * 60)

# --- Boolean think (qwen-style) ---
print("Test 1: Boolean think=True")
print("-" * 60)

response = client.chat.completions.create(
    model=LARGE_MODEL,
    messages=[{"role": "user", "content": "What is 15% of 847?"}],
    extra_body={"think": True},
    max_tokens=1024,
)

choice = response.choices[0]
# The SDK may expose reasoning_content as an extra field
msg = choice.message
if hasattr(msg, "reasoning_content") and msg.reasoning_content:
    thinking = msg.reasoning_content
    print(f"Thinking: {thinking[:200]}...")
    print()
else:
    # Some SDK versions don't parse the field — check raw
    raw = msg.model_dump() if hasattr(msg, "model_dump") else {}
    if raw.get("reasoning_content"):
        print(f"Thinking: {raw['reasoning_content'][:200]}...")
        print()
    else:
        print("(No separate reasoning_content field returned)")
        print()

print(f"Answer:   {msg.content}")
print(f"Tokens:   {response.usage.total_tokens}")

# --- String think for GPT-OSS (reasoning effort) ---
print()
print("Test 2: String think='medium' (GPT-OSS reasoning effort)")
print("-" * 60)

response = client.chat.completions.create(
    model=LARGE_MODEL,
    messages=[{"role": "user", "content": "What is 23 * 47? Show your work."}],
    extra_body={"think": "medium"},
    max_tokens=1024,
)

print(f"Answer:   {response.choices[0].message.content}")
print(f"Tokens:   {response.usage.total_tokens}")
