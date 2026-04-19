#!/usr/bin/env python3
"""
15 - Structured Output (JSON Mode)
====================================

Demonstrates requesting structured JSON output from MindRouter.
Covers:

  - JSON object mode (``response_format={"type": "json_object"}``)
  - JSON schema mode (``response_format={"type": "json_schema", ...}``)
  - Handling 422 errors when thinking mode consumes all tokens

Blog section: "Structured Output and JSON Mode"

Usage:
    python 15_structured_output.py
"""

import json
import openai
from openai import OpenAI
from config import BASE_URL, API_KEY, LARGE_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- JSON Object Mode ---
print(f"Model: {LARGE_MODEL}")
print("=" * 60)
print("Test 1: JSON Object Mode")
print("-" * 60)

response = client.chat.completions.create(
    model=LARGE_MODEL,
    messages=[
        {
            "role": "system",
            "content": "Return a JSON object with fields: name, age, city.",
        },
        {
            "role": "user",
            "content": "Tell me about Alice who is 30 and lives in Portland.",
        },
    ],
    response_format={"type": "json_object"},
    max_tokens=128,
)

content = response.choices[0].message.content
print(f"Raw:    {content}")
data = json.loads(content)
print(f"Parsed: {json.dumps(data, indent=2)}")

# --- JSON Schema Mode ---
print()
print("Test 2: JSON Schema Mode")
print("-" * 60)

response = client.chat.completions.create(
    model=LARGE_MODEL,
    messages=[
        {
            "role": "user",
            "content": "Extract: Bob is 25 and lives in Seattle.",
        },
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age", "city"],
            },
        },
    },
    max_tokens=128,
)

content = response.choices[0].message.content
data = json.loads(content)
print(f"Parsed: {json.dumps(data, indent=2)}")

# --- 422 edge case ---
print()
print("Test 3: 422 edge case (thinking + structured output + low max_tokens)")
print("-" * 60)

try:
    response = client.chat.completions.create(
        model=LARGE_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Solve: what is the integral of x^2 from 0 to 5?",
            },
        ],
        response_format={"type": "json_object"},
        extra_body={"think": True},
        max_tokens=32,  # intentionally low to trigger 422
    )
    content = response.choices[0].message.content
    print(f"  Got response (no 422): {content[:100]}")
except openai.UnprocessableEntityError as e:
    print(f"  Caught 422: {e.message}")
    print("  Fix: increase max_tokens or disable thinking mode")
except Exception as e:
    print(f"  Got {type(e).__name__} instead of 422: {e}")
    print("  (The 422 is hard to trigger reliably; it depends on model behavior)")
