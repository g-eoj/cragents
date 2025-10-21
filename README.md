# `cragents`

**C**onstrain **R**easoning **Agents** to limit reasoning output.

## Why

> And I'm thinking While I'm thinking... (Crackerman, Stone Temple Pilots, 1992)

Reasoning models use a lot of tokens for their reasoning output.
This is resource intensive while not necessarily improving accuracy - have you ever seen a reasoning model talk itself out of the right answer?
So it may be desirable to limit the tokens used.
Doing so can:

- Improved response speed
- Decrease GPU memory requirements
- Provide more space in the context for stuff that matters
- Improve accuracy on user queries that do not require extended analysis

## How

`cragents` provides tools for `pydantic-ai` [agents](https://ai.pydantic.dev/agents/) that will limit the number of paragraphs and the number of sentences per paragraph in reasoning output.
The limits are configurable.

```py
import cragents
from pydantic_ai import Agent

# define an agent as you normally would
agent = Agent(
  ...
)

# constrain reasoning as appropriate
await cragents.constrain_reasoning(
  agent,
  reasoning_paragraph_limit=1,
  reasoning_sentence_limit=8,
)

# call the agent as you normally would
run = await agent.run(...)
```

### Limitations

- Only models that use the `<think></think>` tokens to denote reasoning will work
- vLLM must be started without a reasoning parser (`pydantic-ai` will still extract reasoning content correctly)
