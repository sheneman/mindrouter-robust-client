#!/usr/bin/env python3
"""
14 - Circuit Breaker
====================

Demonstrates the circuit breaker pattern to prevent hammering a failing
cluster. After consecutive failures, the breaker "opens" and fails fast
without sending requests, then periodically "probes" to check recovery.

Three states:
  - CLOSED: normal operation, requests pass through
  - OPEN: failing fast, no requests sent
  - HALF_OPEN: testing if service recovered (one probe request)

Blog section: "Circuit Breakers"

Usage:
    python 14_circuit_breaker.py
"""

import time
import openai
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    """
    Circuit breaker for OpenAI-compatible API calls.

    Opens after ``failure_threshold`` consecutive failures.
    Resets after ``reset_timeout`` seconds (probes with one request).
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"

    def before_request(self):
        if self.state == "open":
            if time.monotonic() - self.last_failure_time >= self.reset_timeout:
                self.state = "half_open"
                print(f"  [CIRCUIT] half_open — probing...")
            else:
                remaining = self.reset_timeout - (time.monotonic() - self.last_failure_time)
                raise CircuitOpenError(
                    f"Circuit is OPEN. Retry in {remaining:.0f}s"
                )

    def on_success(self):
        if self.state != "closed":
            print(f"  [CIRCUIT] closed — recovered!")
        self.failures = 0
        self.state = "closed"

    def on_failure(self):
        self.failures += 1
        self.last_failure_time = time.monotonic()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            print(f"  [CIRCUIT] OPEN after {self.failures} consecutive failures")


def safe_chat(client, breaker, messages, model=CHAT_MODEL):
    """Chat request protected by circuit breaker."""
    breaker.before_request()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=32,
        )
        breaker.on_success()
        return response
    except (openai.APIConnectionError, openai.InternalServerError) as e:
        breaker.on_failure()
        raise
    except openai.RateLimitError:
        # Rate limits are NOT server failures — don't trip the breaker
        raise


# --- Demo ---
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0)
breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10.0)

print(f"Model: {CHAT_MODEL}")
print("Circuit breaker: threshold=3, reset=10s")
print("-" * 60)

# Test 1: Successful request through breaker
print("\nTest 1: Valid request through circuit breaker")
try:
    response = safe_chat(
        client, breaker,
        [{"role": "user", "content": "Say hi."}],
    )
    print(f"  OK: {response.choices[0].message.content}")
    print(f"  State: {breaker.state}, failures: {breaker.failures}")
except Exception as e:
    print(f"  Error: {e}")

# Test 2: Simulate failures by using a bad model
print("\nTest 2: Triggering circuit breaker with bad requests")
bad_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5.0)
for i in range(5):
    try:
        # Use a non-existent model that returns 404 (not a server error)
        # Instead, simulate by calling with the real client
        # For demo purposes, show the breaker state transitions
        print(f"  Attempt {i+1}: state={bad_breaker.state}, failures={bad_breaker.failures}")
        bad_breaker.before_request()
        # Simulate a failure
        bad_breaker.on_failure()
    except CircuitOpenError as e:
        print(f"  BLOCKED: {e}")

print(f"\n  Final state: {bad_breaker.state}")
print(f"  The breaker prevents further requests until reset_timeout elapses.")
