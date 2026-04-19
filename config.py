"""
Shared configuration for all MindRouter example scripts.

Reads settings from environment variables or a .env file.
Every script imports this module to get a consistent client setup.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Load .env file if present (no third-party dependency required)
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# ---------------------------------------------------------------------------
# Required settings
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("MINDROUTER_BASE_URL", "")
API_KEY = os.environ.get("MINDROUTER_API_KEY", "")

if not BASE_URL:
    print("ERROR: Set MINDROUTER_BASE_URL in your environment or .env file.")
    print("  cp .env.example .env   # then edit .env")
    sys.exit(1)

if not API_KEY:
    print("ERROR: Set MINDROUTER_API_KEY in your environment or .env file.")
    print("  cp .env.example .env   # then edit .env")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Default models (override via env vars)
# ---------------------------------------------------------------------------
CHAT_MODEL = os.environ.get("MINDROUTER_CHAT_MODEL", "microsoft/phi-4")
LARGE_MODEL = os.environ.get("MINDROUTER_LARGE_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.environ.get("MINDROUTER_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
