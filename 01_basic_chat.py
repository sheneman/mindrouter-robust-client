#!/usr/bin/env python3
"""
01 - Basic Chat Completion
==========================

Demonstrates the simplest possible chat completion call against MindRouter
using the OpenAI Python SDK. Covers:

  - Creating a sync client pointed at MindRouter
  - Sending a chat completion request
  - Reading the response content, finish_reason, and token usage

Blog section: "Getting Started" and "Chat Completions"

Usage:
    python 01_basic_chat.py
"""

from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

print(f"Model: {CHAT_MODEL}")
print("-" * 60)

response = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "Explain quantum entanglement in two sentences."},
    ],
    max_tokens=256,
    temperature=0.7,
)

choice = response.choices[0]
print(f"Response:       {choice.message.content}")
print(f"Finish reason:  {choice.finish_reason}")
print(f"Prompt tokens:  {response.usage.prompt_tokens}")
print(f"Compl. tokens:  {response.usage.completion_tokens}")
print(f"Total tokens:   {response.usage.total_tokens}")
