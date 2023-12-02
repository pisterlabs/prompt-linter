from steamship import Steamship #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.base.mime_types import MimeTypes #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)
from typing import Any, List, Union
from steamship import Block, Task #upm package(steamship)
from steamship.agents.schema import AgentContext, Tool #upm package(steamship)
from steamship.agents.utils import get_llm, with_llm #upm package(steamship)

PLUGIN_HANDLE = "duckduckgo-wrapper"

class DuckDuckGoTool(Tool):
    """
    Tool for searching the web
    """

    name: str = "DuckDuckGoTool"
    human_description: str = "Useful for searching the web for information."
    agent_description: str = ("Used to search the web for new information.")

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        search = context.client.use_plugin(PLUGIN_HANDLE)

        blocks = []
        for block in tool_input:
        # If the block is not text, simply pass it through.
            if not block.is_text():
                continue
            task = search.tag(doc=block.text)
            task.wait()
            result = task.output.file.blocks[0].tags[0].value
            blocks.append(Block(text=result["result"],mime_type=MimeTypes.TXT))

        return blocks
        
if __name__ == "__main__":
    with Steamship.temporary_workspace() as client:
        ToolREPL(DuckDuckGoTool()).run_with_client(
            client=client, context=with_llm(llm=OpenAI(client=client, temperature=0.9))
        )