#!/usr/bin/env python3
"""
08 - Retry with Tenacity
========================

Demonstrates using the ``tenacity`` library for fine-grained retry control:
custom backoff curves, retry-on-specific-errors, and logging between retries.

Blog section: "Retry Strategies" (Level 2)

Requirements:
    pip install tenacity

Usage:
    python 08_retry_tenacity.py
"""

import logging
import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from config import BASE_URL, API_KEY, CHAT_MODEL

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def resilient_chat(client, **kwargs):
    """Chat completion with tenacity-managed retries."""
    return client.chat.completions.create(**kwargs)


# Disable SDK-level retries so tenacity controls everything
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)

print(f"Model: {CHAT_MODEL}")
print("Strategy: tenacity — random exponential backoff, 6 attempts max")
print("-" * 60)

response = resilient_chat(
    client,
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "Name three planets. Be brief."}],
    max_tokens=64,
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens:   {response.usage.total_tokens}")
