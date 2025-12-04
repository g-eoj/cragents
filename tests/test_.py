import pytest
from inline_snapshot import snapshot
from pydantic_ai import ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cragents import CRAgent, vllm_model_profile

pytestmark = pytest.mark.anyio


model = OpenAIChatModel(
    model_name="...",
    provider=OpenAIProvider(api_key="...", base_url="..."),
    profile=vllm_model_profile,
)


async def test_default_agent_output():
    agent = CRAgent(model)
    await agent.constrain_reasoning(1, 2)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "guided_grammar": """\
start: <think> reason </think> <tool_call> tool_call </tool_call>
reason: paragraph{1,1}
paragraph: NL sentence{1,2} NL
sentence[lazy]: /[^\\.\\n]+/ (".")
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"type": "string"}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
NL: /\\n/
Q: /"/
""",
            }
        }
    )


async def test_deduplicate_output_type():
    agent = CRAgent(model, output_type=[ToolOutput(bool, name="one"), ToolOutput(bool, name="two")])
    await agent.constrain_reasoning(1, 2)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "guided_grammar": """\
start: <think> reason </think> <tool_call> tool_call </tool_call>
reason: paragraph{1,1}
paragraph: NL sentence{1,2} NL
sentence[lazy]: /[^\\.\\n]+/ (".")
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
NL: /\\n/
Q: /"/
""",
            }
        }
    )


async def test_multiple_tool_outputs():
    agent = CRAgent(model, output_type=[ToolOutput(bool), ToolOutput(int)])
    await agent.constrain_reasoning(1, 2)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "guided_grammar": """\
start: <think> reason </think> <tool_call> tool_call </tool_call>
reason: paragraph{1,1}
paragraph: NL sentence{1,2} NL
sentence[lazy]: /[^\\.\\n]+/ (".")
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"anyOf": [{"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}, {"properties": {"response": {"type": "integer"}}, "required": ["response"], "type": "object"}]}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
NL: /\\n/
Q: /"/
""",
            }
        }
    )


async def test_mixed_output_type():
    agent = CRAgent(model, output_type=[ToolOutput(bool), str])
    await agent.constrain_reasoning(1, 2)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "guided_grammar": """\
start: <think> reason </think> <tool_call> tool_call </tool_call>
reason: paragraph{1,1}
paragraph: NL sentence{1,2} NL
sentence[lazy]: /[^\\.\\n]+/ (".")
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"anyOf": [{"type": "string"}, {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}]}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
NL: /\\n/
Q: /"/
""",
            }
        }
    )
