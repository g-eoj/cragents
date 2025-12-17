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


import dataclasses
import json
import re
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai import BinaryImage, DeferredToolRequests, _output, _utils, output

JsonSchema = dict[str, Any]


@dataclasses.dataclass
class Anchor:
    text: str


@dataclasses.dataclass
class Block:
    max_newlines: int
    max_char_captures: int
    chars_to_capture: str = "."


@dataclasses.dataclass
class Free:
    pass


@dataclasses.dataclass
class Think:
    guide: list[Anchor | Block | Free]
    start_token: str = "<think>"
    stop_token: str = "</think>"


@dataclasses.dataclass
class Tools:
    json_schema: JsonSchema | None = None
    tool_name_regex: str = "/[a-zA-Z0-9_]+/"
    tool_names: list[str] | None = None
    start_token: str = "<tool_call>"
    stop_token: str = "</tool_call>"


def build_json_schema(output_schema: _output.OutputSchema[output.OutputDataT]) -> JsonSchema:
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


def build_grammar(guide: list[Free | Think | Tools]) -> str:
    start_def = "start: "
    custom_defs: list[str] = []
    default_defs = [
        "FREE: /.*/",
        "NL: /\\n/",
        'Q: /"/',
    ]

    for x in guide:
        if isinstance(x, Think):
            start_def += f"{x.start_token} NL "

            for i, y in enumerate(x.guide):
                if isinstance(y, Anchor):
                    start_def += f'"{y.text}" '

                if isinstance(y, Block):
                    block_uid = f"block_{i}"
                    p_uid = f"p_{i}"
                    s_uid = f"s_{i}"

                    start_def += f"{block_uid} "

                    custom_defs.append(f"{block_uid}: {p_uid}{{1,{y.max_newlines}}}")
                    custom_defs.append(f"{p_uid}: {s_uid}{{1,{y.max_char_captures}}} NL NL")
                    custom_defs.append(
                        f'{s_uid}[lazy]: /[^{re.escape(y.chars_to_capture)}\\n]+/ ("{y.chars_to_capture}")'
                    )

                if isinstance(y, Free):
                    start_def += "FREE "

            start_def += f"{x.stop_token} "

        if isinstance(x, Tools):
            start_def += f"{x.start_token} tool_call {x.stop_token}"

            custom_defs.append(
                'tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"'
            )
            custom_defs.append("tool_schema: %json " + json.dumps(x.json_schema))
            if not x.tool_names:
                custom_defs.append(f"FUNCTION_NAME: {x.tool_name_regex}")
            else:
                tool_names = [f'"{tool_name}"' for tool_name in x.tool_names]
                custom_defs.append(f"FUNCTION_NAME: ({' | '.join(tool_names)})")

        if isinstance(x, Free):
            start_def += "FREE "

    grammar = "\n".join([start_def] + custom_defs + default_defs)
    return grammar


def make_guided_extra_body(guide: list[Free | Think | Tools]) -> dict[str, Any]:
    grammar = build_grammar(guide)
    extra_body = {
        "chat_template_kwargs": {
            "add_generation_prompt": False,
            "enable_thinking": False,
        },
        "guided_grammar": grammar,
    }
    return extra_body
