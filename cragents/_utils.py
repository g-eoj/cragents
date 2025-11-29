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

from pydantic import TypeAdapter
from pydantic_ai import BinaryImage, DeferredToolRequests, RunContext, _output, _utils
from pydantic_ai.profiles import JsonSchemaTransformer
from pydantic_ai.toolsets import AbstractToolset

JsonSchema = dict[str, Any]


class InlineDefJsonSchemaTransformer(JsonSchemaTransformer):
    # does llguidance require inline defs?
    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, prefer_inlined_defs=True, strict=strict)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


async def get_toolset_schemas(ctx: RunContext, toolset: AbstractToolset) -> list[JsonSchema]:
    schemas: list[JsonSchema] = []
    tools = await toolset.get_tools(ctx)
    for tool in tools.values():
        schema = tool.tool_def.parameters_json_schema
        schemas.append(schema)
    return schemas


def build_json_schema(output_schema: _output.OutputSchema) -> JsonSchema:
    # allow any output with {'type': 'string'} if no constraints
    if not any(
        [
            output_schema.allows_deferred_tools,
            output_schema.allows_image,
            output_schema.object_def,
            output_schema.toolset,
        ]
    ):
        return TypeAdapter(str).json_schema()

    json_schemas: list[JsonSchema] = []

    processor = getattr(output_schema, "processor", None)
    if isinstance(processor, _output.ObjectOutputProcessor):
        json_schema = processor.object_def.json_schema
        json_schemas.append(json_schema)

    elif output_schema.toolset:
        if output_schema.allows_text:
            json_schema = TypeAdapter(str).json_schema()
            json_schemas.append(json_schema)
        for tool_processor in output_schema.toolset.processors.values():
            json_schema = tool_processor.object_def.json_schema
            if json_schema not in json_schemas:
                json_schemas.append(json_schema)

    elif output_schema.allows_text:
        json_schema = TypeAdapter(str).json_schema()
        json_schemas.append(json_schema)

    if output_schema.allows_deferred_tools:
        json_schema = TypeAdapter(DeferredToolRequests).json_schema(mode="serialization")
        if json_schema not in json_schemas:
            json_schemas.append(json_schema)

    if output_schema.allows_image:
        json_schema = TypeAdapter(BinaryImage).json_schema()
        json_schema = {k: v for k, v in json_schema["properties"].items() if k in ["data", "media_type"]}
        if json_schema not in json_schemas:
            json_schemas.append(json_schema)

    if len(json_schemas) == 1:
        return json_schemas[0]

    json_schemas, all_defs = _utils.merge_json_schema_defs(json_schemas)
    json_schema: JsonSchema = {"anyOf": json_schemas}
    if all_defs:
        json_schema["$defs"] = all_defs

    return json_schema


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
        "FUNCTION_NAME: /[a-zA-Z0-9_]+/\n"
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
