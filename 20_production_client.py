#!/usr/bin/env python3
"""
20 - Production Client (All Patterns Combined)
================================================

A complete, production-ready async client that combines:
  - Adaptive concurrency (reduces on 429/503, recovers on success)
  - Exponential backoff with full jitter
  - Circuit breaker (stops requests when cluster is down)
  - Client-side RPM throttling
  - Usage tracking
  - Proper connection management

This is the "putting it all together" script from the blog post.

Blog section: "Putting It All Together: A Production Client"

Usage:
    python 20_production_client.py
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

import httpx
import openai
from openai import AsyncOpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    pass


@dataclass
class UsageStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    retries: int = 0
    rate_limits: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class MindRouterClient:
    """
    Production-grade async client for MindRouter.

    Features:
    - Adaptive concurrency (reduces on 429/503, recovers on success)
    - Exponential backoff with full jitter
    - Circuit breaker (stops requests when cluster is down)
    - Client-side RPM throttling
    - Usage tracking
    - Proper connection management
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_concurrency: int = 10,
        min_concurrency: int = 1,
        rpm_limit: int = 0,
        max_retries: int = 4,
        circuit_threshold: int = 5,
        circuit_reset: float = 30.0,
        timeout: float = 200.0,
    ):
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_concurrency * 2,
                max_keepalive_connections=max_concurrency,
            ),
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        self._concurrency = max_concurrency
        self._min_concurrency = min_concurrency
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()

        self._rpm_interval = 60.0 / rpm_limit if rpm_limit > 0 else 0
        self._last_request_time = 0.0
        self._rpm_lock = asyncio.Lock()

        self._max_retries = max_retries

        self._circuit_failures = 0
        self._circuit_threshold = circuit_threshold
        self._circuit_reset = circuit_reset
        self._circuit_opened_at = 0.0
        self._circuit_state = "closed"

        self._successes_since_reduction = 0
        self._last_reduction_time = 0.0

        self.stats = UsageStats()

    async def close(self):
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # -- Circuit breaker --

    def _check_circuit(self):
        if self._circuit_state == "open":
            if time.monotonic() - self._circuit_opened_at >= self._circuit_reset:
                self._circuit_state = "half_open"
            else:
                remaining = self._circuit_reset - (time.monotonic() - self._circuit_opened_at)
                raise CircuitOpenError(f"Circuit OPEN, retry in {remaining:.0f}s")

    def _circuit_success(self):
        self._circuit_failures = 0
        self._circuit_state = "closed"

    def _circuit_failure(self):
        self._circuit_failures += 1
        if self._circuit_failures >= self._circuit_threshold:
            self._circuit_state = "open"
            self._circuit_opened_at = time.monotonic()
            logger.warning(f"Circuit OPEN after {self._circuit_failures} failures")

    # -- Concurrency --

    async def _reduce_concurrency(self, factor: float = 0.5):
        async with self._lock:
            new = max(self._min_concurrency, int(self._concurrency * factor))
            if new < self._concurrency:
                logger.info(f"Concurrency: {self._concurrency} -> {new}")
                self._concurrency = new
                self._semaphore = asyncio.Semaphore(new)
                self._successes_since_reduction = 0
                self._last_reduction_time = time.monotonic()

    async def _maybe_recover(self):
        async with self._lock:
            self._successes_since_reduction += 1
            elapsed = time.monotonic() - self._last_reduction_time
            if (
                elapsed > 30.0
                and self._successes_since_reduction >= 10
                and self._concurrency < self._max_concurrency
            ):
                new = min(self._concurrency + 2, self._max_concurrency)
                self._concurrency = new
                self._semaphore = asyncio.Semaphore(new)
                self._successes_since_reduction = 0

    # -- RPM throttling --

    async def _throttle(self):
        if self._rpm_interval <= 0:
            return
        async with self._rpm_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._rpm_interval:
                await asyncio.sleep(self._rpm_interval - elapsed)
            self._last_request_time = time.monotonic()

    # -- Core request --

    async def chat(self, messages: list, model: str = CHAT_MODEL, **kwargs):
        self._check_circuit()
        delay = 1.0
        last_error = None

        for attempt in range(self._max_retries + 1):
            await self._throttle()
            async with self._semaphore:
                try:
                    response = await self.client.chat.completions.create(
                        model=model, messages=messages, **kwargs,
                    )
                    self._circuit_success()
                    await self._maybe_recover()
                    self.stats.requests += 1
                    if response.usage:
                        self.stats.prompt_tokens += response.usage.prompt_tokens
                        self.stats.completion_tokens += response.usage.completion_tokens
                    return response

                except openai.RateLimitError as e:
                    self.stats.rate_limits += 1
                    last_error = e
                    await self._reduce_concurrency(0.5)
                    await asyncio.sleep(random.uniform(0, delay))
                    delay = min(delay * 2, 60.0)

                except openai.InternalServerError as e:
                    last_error = e
                    self.stats.retries += 1
                    if e.status_code == 503:
                        await self._reduce_concurrency(0.75)
                    self._circuit_failure()
                    await asyncio.sleep(random.uniform(0, delay))
                    delay = min(delay * 2, 60.0)

                except (openai.APIConnectionError, openai.APITimeoutError) as e:
                    last_error = e
                    self.stats.retries += 1
                    self._circuit_failure()
                    await asyncio.sleep(random.uniform(0, delay))
                    delay = min(delay * 2, 60.0)

                except (
                    openai.BadRequestError,
                    openai.AuthenticationError,
                    openai.PermissionDeniedError,
                    openai.NotFoundError,
                    openai.UnprocessableEntityError,
                ):
                    self.stats.errors += 1
                    raise

        self.stats.errors += 1
        raise last_error or RuntimeError("All retries exhausted")


async def main():
    print(f"Model: {CHAT_MODEL}")
    print("=" * 60)

    async with MindRouterClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        max_concurrency=5,
        rpm_limit=0,  # no client-side RPM limit for this demo
    ) as mr:
        # Single request
        print("Single request:")
        response = await mr.chat(
            messages=[{"role": "user", "content": "Hello! Say hi back in one word."}],
            max_tokens=16,
        )
        print(f"  {response.choices[0].message.content}")

        # Batch processing
        print(f"\nBatch processing (15 prompts, max concurrency=5):")
        prompts = [f"What is {i} * 3? Answer with just the number." for i in range(15)]

        t0 = time.monotonic()
        tasks = [
            mr.chat([{"role": "user", "content": p}], max_tokens=16)
            for p in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.monotonic() - t0

        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"  [{i:2d}] FAIL: {r}")
            else:
                print(f"  [{i:2d}] {r.choices[0].message.content.strip()}")

        print(f"\n{'=' * 60}")
        print(f"Success: {len(successes)}, Failed: {len(failures)}")
        print(f"Wall time: {wall_time:.1f}s")
        print(f"Total tokens: {mr.stats.total_tokens:,}")
        print(f"Rate limits hit: {mr.stats.rate_limits}")
        print(f"Retries: {mr.stats.retries}")
        print(f"Errors: {mr.stats.errors}")


if __name__ == "__main__":
    asyncio.run(main())
