# MindRouter Client Examples

Production-tested Python examples for building robust, resilient client code against [MindRouter](https://github.com/ui-insight/MindRouter)'s OpenAI-compatible API.

These scripts accompany the blog post *"Writing Robust, Resilient Python Clients for MindRouter"* and are designed to be standalone, runnable, and educational. Every script has been tested against a live MindRouter v2.4.0+ cluster.

<img width="1024" height="576" alt="8b1a1fa4f29643a74d104ad46863446290b1009413134864d42aacd49a52dac1_786bf5dc" src="https://github.com/user-attachments/assets/41bcd63e-e81e-49e4-abda-75d98f9047b5" />


## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your connection
cp .env.example .env
# Edit .env with your MindRouter URL and API key

# 3. Run any example
python 01_basic_chat.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MINDROUTER_BASE_URL` | Yes | — | MindRouter base URL (include `/v1`) |
| `MINDROUTER_API_KEY` | Yes | — | Your MindRouter API key |
| `MINDROUTER_CHAT_MODEL` | No | `microsoft/phi-4` | Default model for chat examples |
| `MINDROUTER_LARGE_MODEL` | No | `openai/gpt-oss-120b` | Model for advanced examples (thinking, tools, structured output) |
| `MINDROUTER_EMBED_MODEL` | No | `Qwen/Qwen3-Embedding-8B` | Model for embedding examples |

## Scripts

### Core Endpoints

| Script | Blog Section | What It Demonstrates |
|--------|-------------|---------------------|
| `01_basic_chat.py` | Getting Started, Chat Completions | Simplest possible chat completion — send a message, read the response, check token usage and finish_reason |
| `02_streaming.py` | Streaming Responses | Synchronous streaming with `stream=True` — tokens print as they arrive |
| `03_async_streaming.py` | Streaming Responses (Async) | Async streaming with `AsyncOpenAI` — foundation for concurrent applications |
| `04_embeddings.py` | Embeddings | Single and batch text embeddings for semantic search / RAG |
| `05_model_discovery.py` | Model Discovery | List all available models with MindRouter-specific metadata (capabilities, context length, backends, quantization) |
| `18_rerank.py` | Reranking and Scoring | Document reranking via `/v1/rerank` using httpx (vLLM backends only) |
| `19_tokenize.py` | Tokenization | Count input tokens before sending a request via `/v1/tokenize` |

### Error Handling and Retries

| Script | Blog Section | What It Demonstrates |
|--------|-------------|---------------------|
| `06_error_handling.py` | Error Handling | Exception hierarchy, triggering 404/401 errors, classifying retryable vs fatal errors |
| `07_retry_builtin.py` | Retry Strategies (Level 1) | SDK built-in `max_retries` — simplest retry approach |
| `08_retry_tenacity.py` | Retry Strategies (Level 2) | `tenacity` library — custom backoff curves, retry-on-specific-errors, inter-retry logging |
| `09_retry_manual_backoff.py` | Retry Strategies (Level 3) | Hand-rolled exponential backoff with full jitter — maximum control |

### Rate Limits, Concurrency, and Resilience

| Script | Blog Section | What It Demonstrates |
|--------|-------------|---------------------|
| `10_usage_tracker.py` | Rate Limits (Track Usage) | Cumulative token tracking across requests for quota awareness |
| `11_concurrency_semaphore.py` | Concurrency (Basic) | `asyncio.Semaphore`-bounded concurrent requests — standard batch processing pattern |
| `12_adaptive_concurrency.py` | Concurrency (Adaptive) | Dynamic concurrency adjustment: halve on 429, reduce on 503, recover on sustained success |
| `13_timeouts.py` | Timeouts and Configuration | httpx timeout tuning for self-hosted LLMs (connect/read/write/pool timeouts) |
| `14_circuit_breaker.py` | Circuit Breakers | Circuit breaker pattern: CLOSED → OPEN → HALF_OPEN state machine |

### Advanced Features

| Script | Blog Section | What It Demonstrates |
|--------|-------------|---------------------|
| `15_structured_output.py` | Structured Output / JSON Mode | JSON object mode, JSON schema mode, and the 422 edge case with thinking mode |
| `16_thinking_mode.py` | Thinking / Reasoning Mode | Boolean `think=True` (qwen-style) and string `think="medium"` (GPT-OSS reasoning effort) |
| `17_tool_calling.py` | Tool Calling / Function Calling | Full tool-calling loop: define tools → model requests call → execute → send results back |

### Production Client

| Script | Blog Section | What It Demonstrates |
|--------|-------------|---------------------|
| `20_production_client.py` | Putting It All Together | Complete `MindRouterClient` class combining adaptive concurrency, circuit breaker, RPM throttling, usage tracking, and proper connection management |

## How the Scripts Map to HTTP Status Codes

| Status Code | Scripts That Demonstrate It |
|-------------|----------------------------|
| `200 OK` | All scripts (success path) |
| `401 Unauthorized` | `06_error_handling.py` (bad API key test) |
| `404 Not Found` | `06_error_handling.py` (nonexistent model test) |
| `422 Unprocessable Entity` | `15_structured_output.py` (thinking + structured output + low max_tokens) |
| `429 Too Many Requests` | `12_adaptive_concurrency.py`, `20_production_client.py` (adaptive handling) |
| `502 Bad Gateway` | `09_retry_manual_backoff.py`, `20_production_client.py` (retry logic) |
| `503 Service Unavailable` | `12_adaptive_concurrency.py`, `20_production_client.py` (concurrency reduction) |

## Architecture

```
.env.example          # Template — copy to .env and fill in your values
.env                  # Your credentials (git-ignored)
config.py             # Shared configuration — every script imports this
requirements.txt      # Python dependencies
01_basic_chat.py      # ... through ...
20_production_client.py
```

All scripts import `config.py`, which reads `.env` and provides `BASE_URL`, `API_KEY`, `CHAT_MODEL`, `LARGE_MODEL`, and `EMBED_MODEL` constants.

## Requirements

- Python 3.9+
- `openai` >= 1.30.0
- `httpx` >= 0.27.0
- `tenacity` >= 8.2.0 (only for `08_retry_tenacity.py`)

## License

These examples are provided under the MIT License. Use them as starting points for your own MindRouter client applications.
