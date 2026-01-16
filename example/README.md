# Agent Example

A basic question/answer agent.

## Features

- Web search
- Reads web pages and PDFs
- Returns reference links for web sources
- Can use Python runtime for calculations and simulations

## Requirements

- A linux system with a GPU
- [uv](https://docs.astral.sh/uv/) installed
- A [serper](https://serper.dev) API token for web search

## Quick Start

Set environment variables:

- `HF_TOKEN`
- `SERPER_API_TOKEN`
- `VLLM_API_KEY`
- `VLLM_BASE_URL`
- `VLLM_MODEL_NAME`

Install Firefox for playwright:
```sh
uv run playwright install firefox
```

Launch vLLM with:
```sh
uv run --with vllm==0.13 vllm serve $VLLM_MODEL_NAME --gpu-memory-utilization 0.92 --api-key $VLLM_API_KEY --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 38000
```

Submit a query to the agent:
```sh
uv run python agent.py --query "What is the bore and stroke of a 555 big block Chevy?
```
