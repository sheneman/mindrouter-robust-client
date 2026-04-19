#!/usr/bin/env python3
"""
17 - Tool Calling / Function Calling
======================================

Demonstrates OpenAI-compatible tool calling against MindRouter. The model
can request to call functions, and you provide the results back.

Blog section: "Tool Calling / Function Calling"

Usage:
    python 17_tool_calling.py
"""

import json
from openai import OpenAI
from config import BASE_URL, API_KEY, LARGE_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'San Francisco, CA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
]


def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Fake weather function for demonstration."""
    return {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "unit": unit,
        "condition": "sunny",
    }


print(f"Model: {LARGE_MODEL}")
print("=" * 60)

# Step 1: Send message with tools
print("Step 1: Sending request with tool definitions...")
response = client.chat.completions.create(
    model=LARGE_MODEL,
    messages=[{"role": "user", "content": "What's the weather in Moscow, Idaho?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=256,
)

choice = response.choices[0]
print(f"  Finish reason: {choice.finish_reason}")

if choice.finish_reason == "tool_calls":
    # Step 2: Execute tool calls
    print()
    print("Step 2: Model requested tool calls:")
    for tc in choice.message.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        print(f"  Call: {name}({args})")
        result = get_weather(**args)
        print(f"  Result: {result}")

        # Step 3: Send results back
        print()
        print("Step 3: Sending tool results back to model...")
        followup = client.chat.completions.create(
            model=LARGE_MODEL,
            messages=[
                {"role": "user", "content": "What's the weather in Moscow, Idaho?"},
                choice.message,
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                },
            ],
            tools=tools,
            max_tokens=256,
        )
        print(f"  Final answer: {followup.choices[0].message.content}")
        print(f"  Tokens: {followup.usage.total_tokens}")
else:
    # Model answered directly without calling tools
    print(f"  Direct answer: {choice.message.content}")
