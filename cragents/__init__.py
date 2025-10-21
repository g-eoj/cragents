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


from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile
from typing import Any

from cragents._utils import (
    InlineDefJsonSchemaTransformer,
    get_toolset_schemas,
    make_guided_extra_body,
)


__all__ = (
    #"__version__",
    "constrain_reasoning",
)


async def constrain_reasoning(
    agent: Agent,
    reasoning_paragraph_limit: int,
    reasoning_sentence_limit: int,
    extra_body_override: dict[str, Any] | None = None,
    prepare_model_for_vllm: bool = True
):
    if prepare_model_for_vllm:
        agent.model.profile = OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
            openai_supports_tool_choice_required=False,
            supports_json_object_output=False,
            supports_json_schema_output=True,
        )

    toolsets_schemas = []
    if agent.toolsets is not None:
        for toolset in agent.toolsets:
            toolset_schema = await get_toolset_schemas(toolset)
            # schema can be empty so we need this check
            if toolset_schema:
                toolsets_schemas += toolset_schema

    output_types_schemas = []
    if agent._output_toolset is not None:
        for tool_def in agent._output_toolset._tool_defs:
            schema = tool_def.parameters_json_schema
            schema['title'] = tool_def.name
            schema = InlineDefJsonSchemaTransformer(schema).walk()
            output_types_schemas.append(schema)

    # needs fix for default output type
    final_schema = {
        "anyOf": toolsets_schemas + output_types_schemas
    }

    if agent.model_settings is None:
        agent.model_settings = OpenAIChatModelSettings()

    if extra_body_override is None:
        extra_body = agent.model_settings.get("extra_body", {})
        extra_body.update(
            make_guided_extra_body(
                final_schema,
                reasoning_paragraph_limit=reasoning_paragraph_limit,
                reasoning_sentence_limit=reasoning_sentence_limit,
            )
        )
    else:
        extra_body = extra_body_override
    agent.model_settings["extra_body"] = extra_body
