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


from typing import Any

from pydantic_ai import Agent, RunContext, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles.openai import OpenAIModelProfile

from cragents._utils import (
    JsonSchema,
    build_json_schema,
    get_toolset_schemas,
    make_guided_extra_body,
)
from cragents._version import __version__

__all__ = (
    "__version__",
    "constrain_reasoning",
)


async def constrain_reasoning(
    agent: Agent,
    reasoning_paragraph_limit: int,
    reasoning_sentence_limit: int,
    deps: Any | None = None,
):
    """Limit the number of paragraphs and the number of sentences per paragraph in reasoning output.

    Args:
        agent: a Pydantic AI agent
        reasoning_paragraph_limit: upper bound on the number of paragraphs allowed
        reasoning_sentence_limit: upper bound on the number of sentences allowed per paragraph
        deps: dependencies for Pydantic AI dependency injection system
    """
    agent.model.profile = OpenAIModelProfile(  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        openai_supports_strict_tool_definition=False,
        openai_supports_tool_choice_required=False,
        supports_json_object_output=False,
        supports_json_schema_output=True,
    )

    model: OpenAIChatModel = agent.model  # pyright: ignore[reportAssignmentType]
    ctx = RunContext(deps=deps, model=model, usage=RunUsage())

    json_schema = build_json_schema(agent._output_schema)  # pyright: ignore[report]

    toolsets_schemas: list[JsonSchema] = []
    for toolset in agent.toolsets:
        toolset_schema = await get_toolset_schemas(ctx, toolset)
        # schema can be empty so we need this checkj:
        if toolset_schema:
            toolsets_schemas += toolset_schema

    if toolsets_schemas:
        if "anyOf" in json_schema:
            json_schema["anyOf"] = toolsets_schemas + json_schema["anyOf"]
        else:
            json_schema = {"anyOf": toolsets_schemas + [json_schema]}

    if agent.model_settings is None:
        agent.model_settings = OpenAIChatModelSettings()

    extra_body = agent.model_settings.get("extra_body", {})
    extra_body.update(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        make_guided_extra_body(
            json_schema,
            reasoning_paragraph_limit=reasoning_paragraph_limit,
            reasoning_sentence_limit=reasoning_sentence_limit,
        )
    )
    agent.model_settings["extra_body"] = extra_body
