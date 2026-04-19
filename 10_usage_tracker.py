#!/usr/bin/env python3
"""
10 - Token Usage Tracking
=========================

Demonstrates tracking cumulative token usage across multiple requests
to stay aware of quota consumption. Essential for long-running batch
jobs and quota-conscious applications.

Blog section: "Rate Limits and RPM Quotas" (Track your token usage)

Usage:
    python 10_usage_tracker.py
"""

from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL


class UsageTracker:
    """Track cumulative token usage across requests."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0

    def record(self, usage):
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.request_count += 1

    @property
    def total_tokens(self):
        return self.total_prompt_tokens + self.total_completion_tokens

    def report(self):
        print(f"  Requests:         {self.request_count}")
        print(f"  Prompt tokens:    {self.total_prompt_tokens:,}")
        print(f"  Completion tokens:{self.total_completion_tokens:,}")
        print(f"  Total tokens:     {self.total_tokens:,}")


client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
tracker = UsageTracker()

prompts = [
    "What is gravity? One sentence.",
    "What is photosynthesis? One sentence.",
    "What is DNA? One sentence.",
]

print(f"Model: {CHAT_MODEL}")
print(f"Sending {len(prompts)} requests...")
print("-" * 60)

for i, prompt in enumerate(prompts):
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
    )
    tracker.record(response.usage)
    print(f"  [{i+1}] {response.choices[0].message.content[:80]}...")

print()
print("Cumulative usage:")
tracker.report()
