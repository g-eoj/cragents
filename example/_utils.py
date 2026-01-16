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


from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessage,
    ToolCallPart,
    ToolReturnPart,
)


@dataclass
class LimitDeps:
    """Limit tool calls."""

    limit: int
    _count: int = 0

    def at_limit(self) -> bool:
        if self.limit > self._count:
            self._count += 1
            return False
        else:
            return True


def history_filter(messages: list[ModelMessage]):
    for message in messages:
        for part in message.parts:
            if hasattr(part, "content") and part.content == "Final result processed.":
                part.content = ""
    return messages


def remove_tool_calls(messages: list[ModelMessage]):
    filtered_messages: list[ModelMessage] = []
    for message in messages:
        tool_called = False
        for part in message.parts:
            if isinstance(part, (ToolCallPart, ToolReturnPart)):
                tool_called = True
        if not tool_called:
            filtered_messages.append(message)
    return filtered_messages
