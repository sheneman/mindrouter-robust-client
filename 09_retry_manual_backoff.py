#!/usr/bin/env python3
"""
09 - Manual Exponential Backoff with Jitter
============================================

Demonstrates a hand-rolled retry loop with full jitter. This gives you
maximum control over retry behavior — useful when you need to adapt
based on specific error types (rate limit vs server error).

Blog section: "Retry Strategies" (Level 3) and "Why jitter matters"

Usage:
    python 09_retry_manual_backoff.py
"""

import random
import time
import openai
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL


def chat_with_backoff(
    client: OpenAI,
    messages: list,
    model: str = CHAT_MODEL,
    max_retries: int = 6,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
) -> openai.types.chat.ChatCompletion:
    """
    Chat completion with exponential backoff and full jitter.

    Full jitter (random.uniform(0, delay)) prevents the "thundering herd"
    problem where many clients retry simultaneously after a shared failure.
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except openai.RateLimitError:
            if attempt == max_retries:
                raise
            jittered_delay = random.uniform(0, delay)
            print(f"  Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                  f"Waiting {jittered_delay:.1f}s...")
            time.sleep(jittered_delay)
            delay = min(delay * backoff_factor, max_delay)

        except openai.InternalServerError as e:
            if attempt == max_retries:
                raise
            jittered_delay = random.uniform(0, delay)
            print(f"  Server error {e.status_code} (attempt {attempt + 1}). "
                  f"Waiting {jittered_delay:.1f}s...")
            time.sleep(jittered_delay)
            delay = min(delay * backoff_factor, max_delay)

        except (openai.APIConnectionError, openai.APITimeoutError):
            if attempt == max_retries:
                raise
            jittered_delay = random.uniform(0, delay)
            print(f"  Connection issue (attempt {attempt + 1}). "
                  f"Waiting {jittered_delay:.1f}s...")
            time.sleep(jittered_delay)
            delay = min(delay * backoff_factor, max_delay)


# Disable SDK-level retries
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)

print(f"Model: {CHAT_MODEL}")
print("Strategy: manual exponential backoff with full jitter")
print("-" * 60)

response = chat_with_backoff(
    client,
    messages=[{"role": "user", "content": "What is the speed of light? One sentence."}],
    max_retries=6,
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens:   {response.usage.total_tokens}")
