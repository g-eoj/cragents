# cragents

Grammar-constrained Pydantic AI [agents](https://ai.pydantic.dev/agents/) that think smarter and respond faster.

## Motivation

> And I'm thinking While I'm thinking... (Crackerman, Stone Temple Pilots, 1992)

Reasoning models use a lot of tokens for their reasoning output.
This is resource-intensive while not necessarily improving accuracy.
So it may be desirable to limit the tokens used.
Doing so can:

- Improve response speed
- Decrease GPU memory requirements
- Provide more space in the context for stuff that matters

## Requirements

- Pydantic AI
- The model must be served with vLLM >= 0.13
- vLLM must be started without a reasoning parser

## Example

Guide model output with a composable generation sequence.

1. Start [vLLM](https://vllm.ai/) without a reasoning parser.

```sh
vllm serve $VLLM_MODEL_NAME --gpu-memory-utilization 0.92 --api-key $VLLM_API_KEY --enable-auto-tool-choice --tool-call-parser hermes --max-model-len auto
```

2. Pass the `vllm_model_profile` to a Pydantic AI `OpenAIChatModel`.

```py
import os
from cragents import vllm_model_profile
from pydantic_ai.models.openai import OpenAIChatModel
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

3. Using the model, initialize a `CRAgent` the same as you would for a Pydantic AI [agent](https://ai.pydantic.dev/agents/).

```py
from cragents import CRAgent
from pydantic_ai import ToolOutput

agent = CRAgent(model, output_type=[ToolOutput(bool), ToolOutput(int)])
```

4. Define a generation sequence to guide model output.

```py
from cragents import Anchor, Constrain, Free, Think, UseTools

generation_sequence = [
    Think(
        [
            Anchor("I think "),
            Constrain(max_newlines=1, max_char_captures=1, chars_to_capture=".?!"),
            Anchor("So I should "),
            Free(),
        ]
    ),
    UseTools(),
]

await agent.set_guide(generation_sequence)
```

> Note: You can change the guide at any time by setting it again.

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
            print(part.content)
```

## Primitives

The `set_guide()` method accepts a sequence of elements that control model output. These primitives are reusable and composable. Sequence them in any combination to shape model output.

### Anchor

Force the model to generate exact text.

```py
Anchor(text: str)
```

- `text` - The exact text the model must generate

### Constrain

Limit text expansion. Think of a text block that expands vertically through newlines and horizontally through all other characters.

```py
Constrain(
    max_newlines: int,
    max_char_captures: int,
    chars_to_capture: str = "."
)
```

- `max_newlines` - Upper bound on newlines (vertical expansion)
- `max_char_captures` - Upper bound on capture characters (horizontal expansion)
- `chars_to_capture` - Characters to count for horizontal limiting (default: `"."`)

### Free

Allow unconstrained generation.

```py
Free()
```

> Warning: The model decides when to stop, which may be never.

### UseTools

Force tool call generation.

```py
UseTools(
    json_schema: dict | None = None,
    tool_name_regex: str = "/[a-zA-Z0-9_]+/",
    tool_names: list[str] | None = None,
    start_token: str = "<tool_call>",
    stop_token: str = "</tool_call>"
)
```

- `json_schema` - Schema for allowed tool calls (auto-built from agent config if `None`)
- `tool_name_regex` - Regex pattern for valid tool names
- `tool_names` - Explicit list of allowed tool names
- `start_token` - Token generated before tool calls
- `stop_token` - Token generated after tool calls

### Think (Wrapper)

Wrap a sequence of primitives in reasoning tokens.

```py
Think(
    sequence: Sequence[Anchor | Constrain | Free],
    start_token: str = "<think>",
    stop_token: str = "</think>"
)
```

- `sequence` - Primitives that control the reasoning output
- `start_token` - Token generated before the sequence
- `stop_token` - Token generated after the sequence

> Note: The model will follow whatever guide you provide, but pydantic-ai may not handle all combinations correctly (e.g., tool calls inside think blocks). Use primitives outside the tested patterns at your own risk.
