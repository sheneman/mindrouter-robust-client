#!/usr/bin/env python3
"""
11 - Async Concurrency with Semaphore
======================================

Demonstrates bounded concurrent requests using ``asyncio.Semaphore``.
This is the standard pattern for batch-processing many prompts without
overwhelming the cluster.

Blog section: "Concurrency and Adaptive Throttling" (Basic)

Usage:
    python 11_concurrency_semaphore.py
"""

import asyncio
import time
from openai import AsyncOpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

MAX_CONCURRENT = 5

async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def bounded_chat(prompt: str, index: int) -> dict:
        async with semaphore:
            t0 = time.monotonic()
            response = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
            )
            elapsed = time.monotonic() - t0
            content = response.choices[0].message.content.strip()
            return {"index": index, "content": content, "elapsed": elapsed}

    prompts = [f"What is {i} * {i+1}? Answer with just the number." for i in range(10)]

    print(f"Model: {CHAT_MODEL}")
    print(f"Concurrency: {MAX_CONCURRENT}")
    print(f"Batch size: {len(prompts)}")
    print("-" * 60)

    t0 = time.monotonic()
    tasks = [bounded_chat(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_time = time.monotonic() - t0

    successes = 0
    failures = 0
    for r in results:
        if isinstance(r, Exception):
            failures += 1
            print(f"  FAIL: {r}")
        else:
            successes += 1
            print(f"  [{r['index']:2d}] {r['content']:<20s} ({r['elapsed']:.1f}s)")

    print("-" * 60)
    print(f"Success: {successes}, Failed: {failures}, Wall time: {wall_time:.1f}s")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
