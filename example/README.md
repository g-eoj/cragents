# Agent Example

A basic question/answer agent for use with reasoning models.

## Features

- Web search
- Reads web pages and PDFs
- Returns reference links for web sources
- Can use sandboxed Python runtime for calculations and simulations

## Requirements

- A linux system with a GPU
- [uv](https://docs.astral.sh/uv/) installed
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
uv run --with vllm==0.13 vllm serve $VLLM_MODEL_NAME --gpu-memory-utilization 0.92 --api-key $VLLM_API_KEY --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 38000
```

*Terminal 2* - Prepare virtual environment:
```sh
uv sync --group example
uv run playwright install firefox
```

*Terminal 2* - Submit a query to the agent:
```sh
uv run python agent.py --query "Who did the Warriors play last night?"
```

```sh
09:25:36.684 run graph agent_graph

RouterNode(stimuli='Who did the Warriors play last night?')
09:25:36.687   run node RouterNode
09:25:36.689     router_agent run
09:25:36.692       chat Qwen/Qwen3-VL-8B-Thinking-FP8

ResearchNode(research_query=ResearchQuery(query='Who did the Golden State Warriors play against in their most recent game on January 15, 2026?', reason="The user is asking about the Warriors' opponent from the previous night's game, which would be
January 15, 2026, given the current date is January 16, 2026. The NBA schedule for the future is not known, so I need to find the game details for that date.", source_requirements='NBA official schedule or reputable sports news source',
include_academic_papers=False))
09:25:45.195   run node ResearchNode
09:25:46.210     search_agent run
09:25:46.211       chat Qwen/Qwen3-VL-8B-Thinking-FP8
09:26:08.622     read_agent run
09:26:08.623       chat Qwen/Qwen3-VL-8B-Thinking-FP8

RouterNode(stimuli='<text>The Golden State Warriors played against the New York Knicks on January 15, 2026, in their most recent game. The Warriors won 126-113.</text>\n<reference>https://www.espn.com/nba/recap?gameId=401810439</reference>')
09:26:22.546   run node RouterNode
09:26:22.548     router_agent run
09:26:22.548       chat Qwen/Qwen3-VL-8B-Thinking-FP8
09:26:27.682     approval_agent run
09:26:27.683       chat Qwen/Qwen3-VL-8B-Thinking-FP8

End(data=FinalAnswer(answer='The Golden State Warriors played against the New York Knicks last night, winning 126-113.', references=['https://www.espn.com/nba/recap?gameId=401810439']))
```
