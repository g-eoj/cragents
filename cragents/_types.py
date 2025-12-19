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
from collections.abc import Sequence
from typing import Any

JsonSchema = dict[str, Any]


@dataclasses.dataclass
class Anchor:
    """Force the model to generate this text."""

    text: str


@dataclasses.dataclass
class Constrain:
    """Bound model text output based on newlines and character captures.

    Imagine a text block that expands vertically through newlines and horizontally through all other characters.
    To limit the expansion vertically, we put an upper bound on the number of newlines.
    To limit the expansion horizontally, we put an upper bound on the number of certain characters.

    Args:
        max_newlines: upper bound on the number of newlines the model can generate
        max_char_captures: upper bound on the number of 'capture' characters the model can generate
        chars_to_capture: string of unique characters, by default it is "." as this roughly approximates limiting sentence count
    """

    max_newlines: int
    max_char_captures: int
    chars_to_capture: str = "."


@dataclasses.dataclass
class Free:
    """Allow the model to generate anything.

    WARNING: There is no guarantee free generation will stop and the model will
    continue to the next element in the generation sequence.
    """

    pass


BasicGenerationSequenceElement = Anchor | Constrain | Free


@dataclasses.dataclass
class Think:
    """Force the model to wrap the sequence with 'think' tokens.

    Args:
        sequence: elements that influence model 'reasoning'
        start_token: force the model to generate this token before the sequence starts
        stop_token: force the model to generate this token after the sequence ends
    """

    sequence: Sequence[BasicGenerationSequenceElement]
    start_token: str = "<think>"
    stop_token: str = "</think>"


@dataclasses.dataclass
class UseTools:
    """Force the model to generate tool call(s).

    Note: only the hermes tool parser has been tested.

    Args:
        json_schema: defines the tool calls the model is allowed to generate
        tool_name_regex: use regex to define what tool names the model can select
        tool_names: force the model to choose from these tool names
        start_token: force the model to generate this token before any tool calls
        stop_token: force the model to generate this token after all tool calls
    """

    json_schema: JsonSchema | None = None
    tool_name_regex: str = "/[a-zA-Z0-9_]+/"
    tool_names: list[str] | None = None
    start_token: str = "<tool_call>"
    stop_token: str = "</tool_call>"


GenerationSequenceElement = BasicGenerationSequenceElement | Think | UseTools
