# `cragents`

**C**onstrain **R**easoning **Agents** to limit reasoning output.

## Why

> And I'm thinking While I'm thinking... (Crackerman, Stone Temple Pilots, 1992)

Reasoning models use a lot of tokens for their reasoning output.
This is resource intensive while not necessarily improving accuracy.
So it may be desirable to limit the tokens used.
Doing so can:

- Improved response speed
- Decrease GPU memory requirements
- Provide more space in the context for stuff that matters
- Improve accuracy on user queries that do not require extended analysis

## How

`cragents` provides a utility to constrain `pydantic-ai` [agents](https://ai.pydantic.dev/agents/), if [vLLM](https://docs.vllm.ai/en/stable/) is used to serve the agent's model.
It will limit the number of paragraphs and the number of sentences per paragraph in reasoning output.
The limits are configurable.

```py
import os
from cragents import CRAgent, vllm_model_profile
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    model_name=os.environ["VLLM_MODEL_NAME"],
    provider=OpenAIProvider(
        api_key=os.environ["VLLM_API_KEY"],
        base_url=os.environ["VLLM_BASE_URL"],
    ),
    profile=vllm_model_profile,
    settings=OpenAIChatModelSettings(
        max_tokens=1000,
    ),
)

agent = CRAgent(model)
await agent.constrain_reasoning(reasoning_paragraph_limit=1, reasoning_sentence_limit=1)
run = await agent.run("Hi")
```

Inspecting the `ThinkingPart`s shows that output is constrained.

```py
from pydantic_ai.messages import ThinkingPart

for message in run.all_messages():
    for part in message.parts:
        if isinstance(part, ThinkingPart):
            print(part)
```

```sh
ThinkingPart(content='\nOkay, the user said "Hi".\n', id='content', provider_name='openai')
```

For the above example, vLLM was run on a single RTX 4090:

```sh
uv run vllm serve "Qwen/Qwen3-VL-8B-Thinking-FP8" --gpu-memory-utilization 0.92 --api-key $VLLM_API_KEY --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 40000
```

### Limitations

- Only models that use the `<think></think>` tokens to denote reasoning will work
- Only models that use the `<tool_call></tool_call>` tokens to denote tool calls will work
- vLLM must be started without a reasoning parser (`pydantic-ai` will still extract reasoning content correctly)
