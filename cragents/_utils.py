# Copyright 2025 g-eoj
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from typing import Any

from pydantic_ai import RunContext, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import JsonSchemaTransformer
from pydantic_ai.toolsets import AbstractToolset

JsonSchema = dict[str, Any]


class InlineDefJsonSchemaTransformer(JsonSchemaTransformer):
    # does llguidance require inline defs?
    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, prefer_inlined_defs=True, strict=strict)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


async def get_toolset_schemas(toolset: AbstractToolset) -> list[JsonSchema]:
    ctx = RunContext(
        deps=None,
        model=OpenAIChatModel("_"),
        usage=RunUsage(),
    )
    schemas: list[JsonSchema] = []
    tools = await toolset.get_tools(ctx)
    for name, tool in tools.items():
        schema = tool.tool_def.parameters_json_schema
        schema = InlineDefJsonSchemaTransformer(schema).walk()
        schema["title"] = name
        schemas.append(schema)
    return schemas


def make_guided_extra_body(
    schema: JsonSchema,
    reasoning_paragraph_limit: int,
    reasoning_sentence_limit: int,
):
    guide = (
        f"start: <think> reason </think> <tool_call> tool_call </tool_call>\n"
        f"reason: paragraph{{1,{int(reasoning_paragraph_limit)}}}\n"
        f"paragraph: NL sentence{{1,{int(reasoning_sentence_limit)}}} NL\n"
        f'sentence[lazy]: /[^\\.\\n]+/ (".")\n'
        'tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"\n'
        "tool_schema: %json " + json.dumps(schema) + "\n"
        "FUNCTION_NAME: /[a-zA-Z_]+/\n"
        "NL: /\\n/\n"
        'Q: /"/\n'
    )
    extra_body = {
        "chat_template_kwargs": {
            "add_generation_prompt": False,
            "enable_thinking": False,
        },
        "guided_grammar": guide,
    }
    return extra_body
