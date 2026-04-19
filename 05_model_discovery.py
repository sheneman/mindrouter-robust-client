#!/usr/bin/env python3
"""
05 - Model Discovery
====================

Lists all models available on the MindRouter cluster, including
MindRouter-specific metadata like capabilities, context length,
quantization, and which backends serve each model.

Blog section: "Model Discovery"

Usage:
    python 05_model_discovery.py
"""

import json
import httpx
from openai import OpenAI
from config import BASE_URL, API_KEY

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Basic listing via SDK ---
models = client.models.list()
print(f"Total models available: {len(models.data)}")
print("=" * 80)

for m in sorted(models.data, key=lambda x: x.id):
    print(f"  {m.id}")

# --- Extended metadata via httpx (MindRouter-specific fields) ---
print()
print("=" * 80)
print("Extended metadata (first 5 models):")
print("=" * 80)

response = httpx.get(
    f"{BASE_URL}/models",
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=30.0,
)
data = response.json()

for model in data["data"][:5]:
    print(f"\n  Model: {model['id']}")
    caps = model.get("capabilities", {})
    if caps:
        cap_list = [k for k, v in caps.items() if v]
        print(f"    Capabilities:   {', '.join(cap_list) or 'none'}")
    if model.get("context_length"):
        print(f"    Context length: {model['context_length']:,}")
    if model.get("parameter_count"):
        print(f"    Parameters:     {model['parameter_count']}")
    if model.get("quantization"):
        print(f"    Quantization:   {model['quantization']}")
    backends = model.get("backends", [])
    if backends:
        print(f"    Backends:       {', '.join(backends)}")
