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


import argparse
import asyncio
import os

from rich.console import Console

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from graph import RouterNode, State, agent_graph


async def main():
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("-q", "--query")
    parser.add_argument("-r", "--references_required", default=1, type=int)
    args = parser.parse_args()
    console = Console()
    async with agent_graph.iter(RouterNode(), state=State(task=args.query, references_required=args.references_required)) as run:
        async for node in run:
            console.print(f"\n{node}")

if __name__ == "__main__":
    asyncio.run(main())
