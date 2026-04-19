#!/usr/bin/env python3
"""
12 - Adaptive Concurrency
==========================

Demonstrates dynamically adjusting concurrency based on server feedback:
  - Halves concurrency on 429 (rate limit)
  - Reduces by 25% on 503 (no backend capacity)
  - Gradually increases concurrency after sustained success

This is the most sophisticated concurrency pattern, ideal for batch jobs
that need to maximize throughput without overwhelming the cluster.

Blog section: "Concurrency and Adaptive Throttling" (Adaptive)

Usage:
    python 12_adaptive_concurrency.py
"""

import asyncio
import random
import time
import openai
from openai import AsyncOpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL


class AdaptiveClient:
    """
    Dynamically adjusts concurrency based on server responses.

    - Halves concurrency on 429 (rate limit)
    - Reduces by 25% on 503 (no capacity)
    - Gradually increases concurrency on sustained success
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        initial_concurrency: int = 10,
        min_concurrency: int = 1,
        max_concurrency: int = 50,
        recovery_interval: float = 30.0,
    ):
        self.client = client
        self.concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.recovery_interval = recovery_interval
        self._semaphore = asyncio.Semaphore(initial_concurrency)
        self._lock = asyncio.Lock()
        self._last_reduction = 0.0
        self._consecutive_successes = 0
        self.stats = {"success": 0, "rate_limited": 0, "server_error": 0, "retries": 0}

    async def _adjust_concurrency(self, new_limit: int, reason: str):
        async with self._lock:
            new_limit = max(self.min_concurrency, min(self.max_concurrency, new_limit))
            if new_limit != self.concurrency:
                old = self.concurrency
                self.concurrency = new_limit
                self._semaphore = asyncio.Semaphore(new_limit)
                self._last_reduction = time.monotonic()
                self._consecutive_successes = 0
                print(f"  [ADAPT] Concurrency {old} -> {new_limit} ({reason})")

    async def _maybe_increase(self):
        async with self._lock:
            self._consecutive_successes += 1
            elapsed = time.monotonic() - self._last_reduction
            if (
                elapsed > self.recovery_interval
                and self._consecutive_successes >= 10
                and self.concurrency < self.max_concurrency
            ):
                new = min(self.concurrency + 2, self.max_concurrency)
                old = self.concurrency
                self.concurrency = new
                self._semaphore = asyncio.Semaphore(new)
                self._consecutive_successes = 0
                print(f"  [ADAPT] Concurrency {old} -> {new} (recovery)")

    async def chat(self, messages: list, model: str = CHAT_MODEL, max_retries: int = 4):
        delay = 1.0
        for attempt in range(max_retries):
            async with self._semaphore:
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=32,
                    )
                    await self._maybe_increase()
                    self.stats["success"] += 1
                    return response

                except openai.RateLimitError:
                    self.stats["rate_limited"] += 1
                    self.stats["retries"] += 1
                    await self._adjust_concurrency(self.concurrency // 2, "429 rate limit")
                    await asyncio.sleep(random.uniform(0, delay))
                    delay = min(delay * 2, 60.0)

                except openai.InternalServerError as e:
                    self.stats["server_error"] += 1
                    self.stats["retries"] += 1
                    if e.status_code == 503:
                        await self._adjust_concurrency(
                            int(self.concurrency * 0.75), "503 no capacity"
                        )
                    await asyncio.sleep(random.uniform(0, delay))
                    delay = min(delay * 2, 60.0)

                except (openai.APIConnectionError, openai.APITimeoutError):
                    self.stats["retries"] += 1
                    await asyncio.sleep(random.uniform(0, delay))
                    delay = min(delay * 2, 60.0)

        raise RuntimeError(f"Failed after {max_retries} attempts")


async def main():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)
    adaptive = AdaptiveClient(client, initial_concurrency=5)

    prompts = [f"What is {i} + {i}? Answer with just the number." for i in range(20)]

    print(f"Model: {CHAT_MODEL}")
    print(f"Initial concurrency: 5")
    print(f"Batch size: {len(prompts)}")
    print("-" * 60)

    t0 = time.monotonic()
    tasks = [adaptive.chat([{"role": "user", "content": p}]) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    wall_time = time.monotonic() - t0

    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]

    print("-" * 60)
    print(f"Completed: {len(successes)}, Failed: {len(failures)}")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Final concurrency: {adaptive.concurrency}")
    print(f"Stats: {adaptive.stats}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
