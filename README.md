# cragents

Agents that constrain token generation.

## Motivation

> And I'm thinking While I'm thinking... (Crackerman, Stone Temple Pilots, 1992)

Reasoning models use a lot of tokens for their reasoning output.
This is resource intensive while not necessarily improving accuracy.
So it may be desirable to limit the tokens used.
Doing so can:

- Improved response speed
- Decrease GPU memory requirements
- Provide more space in the context for stuff that matters

## Example

Limit the number of paragraphs and the number of sentences per paragraph in reasoning output.

1. Start [vLLM](https://vllm.ai/) without a reasoning parser.

```sh
vllm serve $VLLM_MODEL_NAME --gpu-memory-utilization 0.92 --api-key $VLLM_API_KEY --enable-auto-tool-choice --tool-call-parser hermes --max-model-len auto
```

2. Pass the `vllm_model_profile` to a Pydantic AI `OpenAIChatModel`.

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
)
```

3. Initialize a `CRAgent` with the model.

```py
agent = CRAgent(model)
```

4. Set the constraints you would like to use.

```py
await agent.constrain_reasoning(reasoning_paragraph_limit=1, reasoning_sentence_limit=1)
```

5. Use the agent as you normally would use a Pydantic AI [agent](https://ai.pydantic.dev/agents/).

```py
run = await agent.run("Hi")
```

Inspecting `ThinkingPart`s should confirm that output is constrained.

```py
from pydantic_ai.messages import ThinkingPart

for message in run.all_messages():
    for part in message.parts:
        if isinstance(part, ThinkingPart):
            print(part)
```

```py
> ThinkingPart(content='\nOkay, the user said "Hi".\n', id='content', provider_name='openai')
```

## Requirements

- The model must be served with vLLM
- vLLM must be started without a reasoning parser
- Models must use `<think></think>` tokens to denote reasoning
- Models must use `<tool_call></tool_call>` tokens to denote tool calls
