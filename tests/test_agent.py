import pytest
from inline_snapshot import snapshot
from pydantic_ai import ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.models.test import TestModel
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


# ── end-to-end set_guide output type tests ────────────────────────────────────


async def test_default_agent_output():
    agent = CRAgent(model)
    await agent.set_guide(generation_sequence)
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
    await agent.set_guide(generation_sequence)
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
    await agent.set_guide(generation_sequence)
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
    await agent.set_guide(generation_sequence)
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


# ── set_guide error handling and model settings ───────────────────────────────


async def test_set_guide_requires_openai_model():
    agent = CRAgent(TestModel())
    with pytest.raises(RuntimeError, match="OpenAIChatModel required"):
        await agent.set_guide([Anchor("hi")])


async def test_set_guide_creates_model_settings_when_none():
    agent = CRAgent(model)
    assert agent.model_settings is None
    await agent.set_guide([Anchor("hi")])
    assert agent.model_settings is not None
    assert "extra_body" in agent.model_settings


async def test_set_guide_preserves_existing_model_settings():
    agent = CRAgent(model, model_settings=OpenAIChatModelSettings(temperature=0.5))
    await agent.set_guide([Anchor("hi")])
    assert agent.model_settings["temperature"] == 0.5
    assert "extra_body" in agent.model_settings


async def test_set_guide_overwrites_on_second_call():
    agent = CRAgent(model)
    await agent.set_guide([Anchor("first")])
    first_grammar = agent.model_settings["extra_body"]["structured_outputs"]["grammar"]
    await agent.set_guide([Anchor("second")])
    second_grammar = agent.model_settings["extra_body"]["structured_outputs"]["grammar"]
    assert first_grammar != second_grammar
    assert "second" in second_grammar


# ── set_guide UseTools schema handling ────────────────────────────────────────


async def test_set_guide_explicit_use_tools_schema_not_overwritten():
    explicit_schema = {"type": "number"}
    agent = CRAgent(model)
    await agent.set_guide([UseTools(json_schema=explicit_schema)])
    grammar = agent.model_settings["extra_body"]["structured_outputs"]["grammar"]
    assert '"type": "number"' in grammar


async def test_set_guide_use_tools_with_registered_tool():
    agent = CRAgent(model)

    @agent.tool_plain
    def my_tool(x: int) -> str:
        return str(x)

    await agent.set_guide([UseTools()])
    grammar = agent.model_settings["extra_body"]["structured_outputs"]["grammar"]
    # The tool's parameter schema (containing "x") should appear in the grammar
    assert '"x"' in grammar


async def test_set_guide_use_tools_tool_names():
    agent = CRAgent(model)
    await agent.set_guide([UseTools(json_schema={"type": "string"}, tool_names=["alpha", "beta"])])
    grammar = agent.model_settings["extra_body"]["structured_outputs"]["grammar"]
    assert 'FUNCTION_NAME: ("alpha" | "beta")' in grammar


async def test_set_guide_merges_toolset_with_anyof_output():
    # When the output schema already has anyOf (multiple output types) and the agent
    # also has registered tools, anyOf = toolset_schemas + return_schema["anyOf"]
    agent = CRAgent(model, output_type=[ToolOutput(bool), ToolOutput(int)])

    @agent.tool_plain
    def helper(x: str) -> str:
        return x

    await agent.set_guide([UseTools()])
    grammar = agent.model_settings["extra_body"]["structured_outputs"]["grammar"]
    assert "tool_schema" in grammar
    assert "anyOf" in grammar


# ── vllm_model_profile ─────────────────────────────────────────────────────────


def test_vllm_profile_strict_tool_definition():
    assert vllm_model_profile.openai_supports_strict_tool_definition is False


def test_vllm_profile_tool_choice_required():
    assert vllm_model_profile.openai_supports_tool_choice_required is False


def test_vllm_profile_json_object_output():
    assert vllm_model_profile.supports_json_object_output is False


def test_vllm_profile_json_schema_output():
    assert vllm_model_profile.supports_json_schema_output is True
