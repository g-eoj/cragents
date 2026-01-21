# Copyright 2026 g-eoj
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


from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import logfire
from cragents import CRAgent, vllm_model_profile
from pydantic_ai import RunContext, ToolDefinition, ToolOutput, format_as_xml
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from _tools import (
    read_url,
    search_papers,
    search_web,
)
from _types import (
    ApprovalResponse,
    CoderResponse,
    CoderTask,
    FinalAnswer,
    Note,
    NoteWithReference,
    PaperSearchResult,
    ResearchQuery,
    SearchResult,
    URLSelection,
)
from _utils import LimitDeps


#logfire.configure(send_to_logfire=False)
#logfire.instrument_pydantic_ai()


model = OpenAIChatModel(
    model_name=os.environ["VLLM_MODEL_NAME"],
    provider=OpenAIProvider(
        api_key=os.environ["VLLM_API_KEY"],
        base_url=os.environ["VLLM_BASE_URL"],
    ),
    profile=vllm_model_profile,
    settings=OpenAIChatModelSettings(
        parallel_tool_calls=False,
    ),
)


@dataclass
class State:
    """Graph state."""

    references_required: int
    task: str
    answer_trys: int = 0
    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    references: list[str] = field(default_factory=list[str])
    research_queries: dict[str, list[NoteWithReference]] = field(default_factory=dict[str, list[NoteWithReference]])


@dataclass
class RouterNode(BaseNode[State]):
    """Routes task steps to tools or agents. Decides when task is complete."""

    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> End[FinalAnswer] | CoderNode | ResearchNode | RouterNode:
        complexity = len(re.split(r"[\.\?\!] ", ctx.state.task)) + 5
        router_agent = CRAgent(
            instructions="Don't call the researcher unnecessarily. However, do call the researcher as many times as needed to collect relevant information.",
            model=model,
            output_type=[
                ToolOutput(CoderTask, name="call_coder"),
                ToolOutput(ResearchQuery, name="call_researcher"),
                ToolOutput(FinalAnswer, name="final_answer"),
            ],
        )
        await router_agent.constrain_reasoning(
            reasoning_paragraph_limit=complexity,
            reasoning_sentence_limit=8,
        )
        if self.feedback is None:
            instructions = f"<query>{ctx.state.task}</query><current_datetime>{datetime.now()}</current_datetime>"
        else:
            instructions = self.feedback
        run = await router_agent.run(
            instructions,
            message_history=ctx.state.messages,
        )
        ctx.state.messages += run.new_messages()

        if isinstance(run.output, CoderTask):
            return CoderNode(run.output)

        if isinstance(run.output, ResearchQuery):
            return ResearchNode(run.output)

        if isinstance(run.output, FinalAnswer):
            ctx.state.answer_trys += 1
            potential_final_answer = run.output
            approval_agent = CRAgent(
                instructions="Make absolutely sure all requirements are met before approving an answer.",
                model=model,
                output_type=[
                    ToolOutput(ApprovalResponse, max_retries=1, name="make_decision"),
                ],
            )
            await approval_agent.constrain_reasoning(
                reasoning_paragraph_limit=complexity,
                reasoning_sentence_limit=8,
            )
            instructions = (
                f"Review the original task: {ctx.state.task}\n\n"
                f"Review the message history and this final answer and decide if it was arrived at correctly: {format_as_xml(potential_final_answer)}\n\n"
            )
            approval_run = await approval_agent.run(
                instructions,
                message_history=ctx.state.messages,
            )
            ctx.state.messages += run.new_messages()
            if approval_run.output.answer_accepted or ctx.state.answer_trys > 3:
                potential_final_answer.references = list(set(ctx.state.references))
                return End(potential_final_answer)

            return RouterNode(feedback=format_as_xml(approval_run.output))


@dataclass
class CoderNode(BaseNode[State, None, CoderTask]):
    """Python coder."""

    task: CoderTask

    @staticmethod
    async def no_comments(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
        tool_defs[0].parameters_json_schema["properties"]["python_code"]["pattern"] = "^[^#]*$"
        return tool_defs

    async def run(self, ctx: GraphRunContext[State]) -> RouterNode:
        complexity = len(re.split(r"[\.\?\!] ", format_as_xml(self.task))) + 1
        python_server = MCPServerStdio("uvx", args=["mcp-run-python@latest", "stdio"], timeout=10)
        python_server = python_server.prepared(prepare_func=CoderNode.no_comments)
        python_server = python_server.filtered(lambda ctx, tool_def: not ctx.deps.at_limit())
        coder_agent = CRAgent(
            model=model,
            deps_type=LimitDeps,
            output_type=[
                ToolOutput(
                    CoderResponse,
                    name="final_result",
                    description="Call this tool when you have completed your task.",
                ),
            ],
            retries=2,
            toolsets=[python_server],
        )
        deps = LimitDeps(5)
        await coder_agent.constrain_reasoning(
            reasoning_paragraph_limit=complexity,
            reasoning_sentence_limit=8,
            deps=deps,
        )
        async with coder_agent:
            run = await coder_agent.run(format_as_xml(self.task), deps=deps)
        return RouterNode(feedback=format_as_xml(run.output))


@dataclass
class ResearchNode(BaseNode[State, None, str]):
    """Finds information."""

    research_query: ResearchQuery

    async def run(self, ctx: GraphRunContext[State]) -> RouterNode | ResearchNode:

        query_hash = hashlib.sha256(format_as_xml(self.research_query).encode()).hexdigest()
        if query_hash in ctx.state.research_queries:
            return RouterNode(feedback=format_as_xml(ctx.state.research_queries[query_hash]))
        ctx.state.research_queries[query_hash] = []

        search_results: list[SearchResult | PaperSearchResult] = await search_web(query=self.research_query.query)
        if self.research_query.include_academic_papers:
            search_results.extend(await search_papers(query=self.research_query.query))

        select_agent = CRAgent(
            model=model,
            output_type=[
                ToolOutput(URLSelection, name="select_url", max_retries=3),
            ],
        )
        await select_agent.constrain_reasoning(
            reasoning_paragraph_limit=8, reasoning_sentence_limit=8,
        )

        read_agent = CRAgent(
            model,
            output_type=[ToolOutput(Note, name="make_note")],
        )

        for _ in range(ctx.state.references_required):
            select_run = await select_agent.run(
                f"Select an URL from the ones below, given the query: {format_as_xml(self.research_query)}\n\n{format_as_xml(search_results)}",
            )
            query_documents = await read_url(query=self.research_query.query, url=select_run.output.url)
            summary_documents = [(i, d) for i, d in zip(query_documents["ids"][0], query_documents["documents"][0])]
            summary_documents.sort(key=lambda x: x[0])
            summary = "\n\n...\n\n".join([x[1] for x in summary_documents])

            await read_agent.constrain_reasoning(
                reasoning_paragraph_limit=8, reasoning_sentence_limit=8
            )
            read_run = await read_agent.run(
                f"Write a note that will help answer:\n\n'{format_as_xml(self.research_query)}'\n\nUse these documents:\n\n{summary}"
            )
            note = NoteWithReference(
                text=read_run.output.text,
                reference=select_run.output.url,
            )
            ctx.state.references.append(note.reference)
            ctx.state.research_queries[query_hash].append(note)
            search_results = [sr for sr in search_results if sr.url != select_run.output.url]

        return RouterNode(feedback=format_as_xml(ctx.state.research_queries[query_hash]))


agent_graph = Graph(nodes=[RouterNode, CoderNode, ResearchNode])
