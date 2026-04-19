#!/usr/bin/env python3
"""
03 - Async Streaming Chat Completion
=====================================

Demonstrates the async variant of streaming using ``AsyncOpenAI``.
This is the foundation for high-concurrency client applications.

Blog section: "Streaming Responses" (Async streaming)

Usage:
    python 03_async_streaming.py
"""

import asyncio
from openai import AsyncOpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    print(f"Model: {CHAT_MODEL}")
    print("-" * 60)

    stream = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Tell me a fun fact about octopuses."}],
        stream=True,
        max_tokens=256,
    )

    full_response = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta

    print()
    print("-" * 60)
    print(f"Total chars: {len(full_response)}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
