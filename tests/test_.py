import pytest
from inline_snapshot import snapshot
from pydantic_ai import ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cragents import Anchor, Constrain, CRAgent, Free, Think, UseTools, vllm_model_profile

pytestmark = pytest.mark.anyio


model = OpenAIChatModel(
    model_name="...",
    provider=OpenAIProvider(api_key="...", base_url="..."),
    profile=vllm_model_profile,
)

generation_sequence = [
    Think([Anchor("I think "), Constrain(1, 2), Anchor("Therefore "), Free()]),
    Anchor("Response: "),
    Constrain(2, 1),
    Free(),
    UseTools(),
]


async def test_default_agent_output():
    agent = CRAgent(model)
    await agent.guide(generation_sequence)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "structured_outputs": {
                    "grammar": """\
start: <think> NL "I think " block_1 "Therefore " FREE </think> "Response: " block_2 FREE <tool_call> tool_call </tool_call>
block_1: p_1{1,1}
p_1: s_1{1,2} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
block_2: p_2{1,2}
p_2: s_2{1,1} NL NL
s_2[lazy]: /[^\\.\\n]+/ ( "." )
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"type": "string"}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
FREE: /[\\S\\s]*/
NL: /\\n/\
"""
                },
            }
        }
    )


async def test_deduplicate_output_type():
    agent = CRAgent(model, output_type=[ToolOutput(bool, name="one"), ToolOutput(bool, name="two")])
    await agent.guide(generation_sequence)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "structured_outputs": {
                    "grammar": """\
start: <think> NL "I think " block_1 "Therefore " FREE </think> "Response: " block_2 FREE <tool_call> tool_call </tool_call>
block_1: p_1{1,1}
p_1: s_1{1,2} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
block_2: p_2{1,2}
p_2: s_2{1,1} NL NL
s_2[lazy]: /[^\\.\\n]+/ ( "." )
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
FREE: /[\\S\\s]*/
NL: /\\n/\
"""
                },
            }
        }
    )


async def test_multiple_tool_outputs():
    agent = CRAgent(model, output_type=[ToolOutput(bool), ToolOutput(int)])
    await agent.guide(generation_sequence)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "structured_outputs": {
                    "grammar": """\
start: <think> NL "I think " block_1 "Therefore " FREE </think> "Response: " block_2 FREE <tool_call> tool_call </tool_call>
block_1: p_1{1,1}
p_1: s_1{1,2} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
block_2: p_2{1,2}
p_2: s_2{1,1} NL NL
s_2[lazy]: /[^\\.\\n]+/ ( "." )
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"anyOf": [{"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}, {"properties": {"response": {"type": "integer"}}, "required": ["response"], "type": "object"}]}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
FREE: /[\\S\\s]*/
NL: /\\n/\
"""
                },
            }
        }
    )


async def test_mixed_output_type():
    agent = CRAgent(model, output_type=[ToolOutput(bool), str])
    await agent.guide(generation_sequence)
    assert agent.model_settings == snapshot(
        {
            "extra_body": {
                "chat_template_kwargs": {"add_generation_prompt": False, "enable_thinking": False},
                "structured_outputs": {
                    "grammar": """\
start: <think> NL "I think " block_1 "Therefore " FREE </think> "Response: " block_2 FREE <tool_call> tool_call </tool_call>
block_1: p_1{1,1}
p_1: s_1{1,2} NL NL
s_1[lazy]: /[^\\.\\n]+/ ( "." )
block_2: p_2{1,2}
p_2: s_2{1,1} NL NL
s_2[lazy]: /[^\\.\\n]+/ ( "." )
tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"
tool_schema: %json {"anyOf": [{"type": "string"}, {"properties": {"response": {"type": "boolean"}}, "required": ["response"], "type": "object"}]}
FUNCTION_NAME: /[a-zA-Z0-9_]+/
FREE: /[\\S\\s]*/
NL: /\\n/\
"""
                },
            }
        }
    )
