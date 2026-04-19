#!/usr/bin/env python3
"""
06 - Error Handling
===================

Demonstrates how to catch and classify OpenAI SDK exceptions when
talking to MindRouter. Covers:

  - The full exception hierarchy (retryable vs fatal)
  - Triggering a 404 (model not found) and inspecting the error body
  - Triggering a 401 (bad API key)
  - Classifying errors for retry decisions

Blog section: "Error Handling"

Usage:
    python 06_error_handling.py
"""

import logging
import openai
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Errors worth retrying
RETRYABLE_ERRORS = (
    openai.RateLimitError,       # 429
    openai.APITimeoutError,      # timeout
    openai.APIConnectionError,   # network
    openai.InternalServerError,  # 500/502/503/504
)

# Errors that will never succeed on retry
FATAL_ERRORS = (
    openai.BadRequestError,          # 400
    openai.AuthenticationError,      # 401
    openai.PermissionDeniedError,    # 403
    openai.NotFoundError,            # 404
    openai.UnprocessableEntityError, # 422
)


def call_with_error_handling(func, *args, **kwargs):
    """Call an OpenAI SDK method with proper error classification."""
    try:
        return func(*args, **kwargs)
    except FATAL_ERRORS as e:
        logger.error(f"Non-retryable error ({e.status_code}): {e.message}")
        return None
    except openai.RateLimitError as e:
        logger.warning(f"Rate limited: {e.message}")
        return None
    except openai.InternalServerError as e:
        logger.warning(f"Server error ({e.status_code}): {e.message}")
        return None
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        logger.warning(f"Connection issue: {e}")
        return None


client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)

# --- Test 1: Valid request ---
print("Test 1: Valid request")
print("-" * 60)
result = call_with_error_handling(
    client.chat.completions.create,
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "Say hello in one word."}],
    max_tokens=16,
)
if result:
    print(f"  OK: {result.choices[0].message.content}")
print()

# --- Test 2: Model not found (404) ---
print("Test 2: Model not found (404)")
print("-" * 60)
try:
    client.chat.completions.create(
        model="nonexistent-model-xyz",
        messages=[{"role": "user", "content": "hello"}],
    )
except openai.NotFoundError as e:
    print(f"  Status:  {e.status_code}")
    print(f"  Message: {e.message}")
    print(f"  Body:    {e.body}")
print()

# --- Test 3: Bad API key (401) ---
print("Test 3: Bad API key (401)")
print("-" * 60)
bad_client = OpenAI(base_url=BASE_URL, api_key="mr2_invalid_key", max_retries=0)
try:
    bad_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "hello"}],
    )
except openai.AuthenticationError as e:
    print(f"  Status:  {e.status_code}")
    print(f"  Message: {e.message}")
print()

# --- Test 4: Error classification demo ---
print("Test 4: Error classification via wrapper")
print("-" * 60)
result = call_with_error_handling(
    client.chat.completions.create,
    model="nonexistent-model-xyz",
    messages=[{"role": "user", "content": "hello"}],
)
print(f"  Returned None (handled gracefully): {result is None}")
