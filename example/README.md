# Agent Example

A basic question/answer agent for use with reasoning models.

## Features

- Web search
- Reads web pages and PDFs
- Returns reference links for web sources
- Option to require a minimum number of references per query
- Can use sandboxed Python runtime for calculations and simulations

## Requirements

- A linux system with a GPU
- [uv](https://docs.astral.sh/uv/) installed
- [deno](https://deno.com/) installed for sandboxed Python runtime
- A [serper](https://serper.dev) API token for web search

## Quick Start

Set environment variables in two terminals:

- `HF_TOKEN` (optional, depending on model choice)
- `SERPER_API_TOKEN` (get from [serper](https://serper.dev))
- `VLLM_API_KEY` (set to anything, can't be empty)
- `VLLM_BASE_URL` (`http://localhost:8000/v1`)
- `VLLM_MODEL_NAME` (`Qwen/Qwen3-VL-8B-Thinking-FP8` recommended)

*Terminal 1* - Launch vLLM with:
```sh
uv run --with vllm==0.13 vllm serve $VLLM_MODEL_NAME --gpu-memory-utilization 0.92 --api-key $VLLM_API_KEY --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 60000
```

*Terminal 2* - Prepare virtual environment:
```sh
uv sync --group example
uv run playwright install firefox
```

*Terminal 2* - Submit a query to the agent:
```sh
uv run python agent.py --query "Who did the Warriors play last night?" --references_required 2
```

