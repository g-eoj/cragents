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
import re
from collections.abc import Sequence

from pydantic import TypeAdapter
from pydantic_ai import BinaryImage, DeferredToolRequests, _output, _utils, output

from ._types import (
    Anchor,
    Constrain,
    Free,
    GenerationSequenceElement,
    JsonSchema,
    Think,
    UseTools,
)


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


def build_grammar(generation_sequence: Sequence[GenerationSequenceElement]) -> str:
    start_def = "start: "
    custom_defs: list[str] = []
    default_defs = [
        "FREE: /[\\S\\s]*/",
        "NL: /\\n/",
    ]

    uid = 0
    for element in generation_sequence:
        if isinstance(element, Anchor):
            start_def += f'"{element.text}" '

        if isinstance(element, Constrain):
            uid += 1
            block_uid = f"block_{uid}"
            p_uid = f"p_{uid}"
            s_uid = f"s_{uid}"

            start_def += f"{block_uid} "

            capture = " | ".join([f'"{x}"' for x in element.chars_to_capture])
            custom_defs.append(f"{block_uid}: {p_uid}{{1,{element.max_newlines}}}")
            custom_defs.append(f"{p_uid}: {s_uid}{{1,{element.max_char_captures}}} NL NL")
            custom_defs.append(f"{s_uid}[lazy]: /[^{re.escape(element.chars_to_capture)}\\n]+/ ( {capture} )")

        if isinstance(element, Free):
            start_def += "FREE "

        if isinstance(element, Think):
            start_def += f"{element.start_token} NL "

            for think_element in element.sequence:
                if isinstance(think_element, Anchor):
                    start_def += f'"{think_element.text}" '

                if isinstance(think_element, Constrain):
                    uid += 1
                    block_uid = f"block_{uid}"
                    p_uid = f"p_{uid}"
                    s_uid = f"s_{uid}"

                    start_def += f"{block_uid} "

                    capture = " | ".join([f'"{x}"' for x in think_element.chars_to_capture])
                    custom_defs.append(f"{block_uid}: {p_uid}{{1,{think_element.max_newlines}}}")
                    custom_defs.append(f"{p_uid}: {s_uid}{{1,{think_element.max_char_captures}}} NL NL")
                    custom_defs.append(
                        f"{s_uid}[lazy]: /[^{re.escape(think_element.chars_to_capture)}\\n]+/ ( {capture} )"
                    )

                if isinstance(think_element, Free):
                    start_def += "FREE "

            start_def += f"{element.stop_token} "

        if isinstance(element, UseTools):
            start_def += f"{element.start_token} tool_call {element.stop_token}"

            custom_defs.append(
                'tool_call: "{\\"name\\": \\"" FUNCTION_NAME "\\", \\"arguments\\": " tool_schema "}\\n"'
            )
            custom_defs.append("tool_schema: %json " + json.dumps(element.json_schema))
            if not element.tool_names:
                custom_defs.append(f"FUNCTION_NAME: {element.tool_name_regex}")
            else:
                tool_names = [f'"{tool_name}"' for tool_name in element.tool_names]
                custom_defs.append(f"FUNCTION_NAME: ({' | '.join(tool_names)})")

    grammar = "\n".join([start_def] + custom_defs + default_defs)
    return grammar


def make_guided_extra_body(
    generation_sequence: Sequence[GenerationSequenceElement],
) -> JsonSchema:
    grammar = build_grammar(generation_sequence)
    extra_body = {
        "chat_template_kwargs": {
            "add_generation_prompt": False,
            "enable_thinking": False,
        },
        "structured_outputs": {"grammar": grammar},
    }
    return extra_body
