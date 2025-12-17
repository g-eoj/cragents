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


import copy

from pydantic_ai import Agent, RunContext, RunUsage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.output import OutputDataT
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

from cragents._utils import (
    Anchor,
    Block,
    Free,
    JsonSchema,
    Think,
    Tools,
    build_json_schema,
    make_guided_extra_body,
)
from cragents._version import __version__

__all__ = ("__version__", "CRAgent", "Anchor", "Block", "Free", "Think", "Tools", "vllm_model_profile")


vllm_model_profile = OpenAIModelProfile(
    openai_supports_strict_tool_definition=False,
    openai_supports_tool_choice_required=False,
    supports_json_object_output=False,
    supports_json_schema_output=True,
)


class CRAgent(Agent[AgentDepsT, OutputDataT]):
    """Pydantic AI Agent with one extra method: `constrain_reasoning`."""

    async def _build_toolset_json_schemas(
        self, ctx: RunContext[AgentDepsT], toolset: AbstractToolset[AgentDepsT]
    ) -> list[JsonSchema]:
        schemas: list[JsonSchema] = []
        tools = await toolset.get_tools(ctx)
        for tool in tools.values():
            schema = tool.tool_def.parameters_json_schema
            schemas.append(schema)
        return schemas

    async def constrain(
        self,
        guide: list[Free | Think | Tools],
        deps: AgentDepsT = None,
    ) -> None:
        """Limit the number of paragraphs and the number of sentences per paragraph in reasoning output.

        Args:
            guide: ...
            deps: dependencies for Pydantic AI dependency injection system
        """
        if not isinstance(self.model, OpenAIChatModel):
            raise RuntimeError("OpenAIChatModel required.")

        processed_guide: list[Free | Think | Tools] = []
        for x in guide:
            x = copy.copy(x)
            if isinstance(x, Tools) and x.json_schema is None:
                return_schema = build_json_schema(self._output_schema)

                toolsets_schemas: list[JsonSchema] = []
                ctx = RunContext(deps=deps, model=self.model, usage=RunUsage())
                for toolset in self.toolsets:
                    # schema can be empty so we need this check
                    if toolset_schema := await self._build_toolset_json_schemas(ctx, toolset):
                        toolsets_schemas += toolset_schema

                if toolsets_schemas:
                    json_schema = {}
                    if "anyOf" in return_schema:
                        json_schema["anyOf"] = toolsets_schemas + return_schema["anyOf"]
                    else:
                        json_schema = {"anyOf": toolsets_schemas + [return_schema]}
                else:
                    json_schema = return_schema
                x.json_schema = json_schema
            processed_guide.append(x)

        extra_body = make_guided_extra_body(processed_guide)

        if self.model_settings is None:
            self.model_settings = OpenAIChatModelSettings()
        self.model_settings["extra_body"] = extra_body
