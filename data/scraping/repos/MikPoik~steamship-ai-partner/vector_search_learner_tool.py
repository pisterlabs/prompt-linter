"""Answers questions with the assistance of a VectorSearch plugin."""
from typing import Any, List, Optional, Union

from steamship import Block, Tag, Task #upm package(steamship)
from steamship.agents.llms import OpenAI #upm package(steamship)
from steamship.agents.schema import AgentContext #upm package(steamship)
from steamship.agents.tools.question_answering.vector_search_tool import VectorSearchTool #upm package(steamship)
from steamship.agents.utils import with_llm #upm package(steamship)
from steamship.utils.repl import ToolREPL #upm package(steamship)


class VectorSearchLearnerTool(VectorSearchTool):
    """Tool to answer questions with the assistance of a vector search plugin."""
    name: str = "VectorSearchLearnerTool"
    human_description: str = "Learns a new fact, event, personal detail or preference of the User and puts it in the Vector Database."
    agent_description: str = (
        "Used to remember a fact, event, personal detail or preference of User."
        "The input is a fact, event, preference or personal detail of User."
        "The output is a confirmation that the fact has been learned."
    )

    def learn_sentence(self, sentence: str, context: AgentContext, metadata: Optional[dict] = None):
        """Learns a sigle sentence-sized piece of text.

        GUIDANCE: No more than about a short sentence is a useful unit of embedding search & lookup.
        """
        index = self.get_embedding_index(context.client)
        tag = Tag(text=sentence, metadata=metadata)
        index.insert(tags=[tag])

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        """Learns a fact with the assistance of an Embedding Index plugin.

        Inputs
        ------
        tool_input: List[Block]
            A list of blocks to be rewritten if text-containing.
        context: AgentContext
            The active AgentContext.

        Output
        ------
        output: List[Blocks]
            A lit of blocks containing the answers.
        """

        output = []
        for input_block in tool_input:
            if input_block.is_text():
                self.learn_sentence(input_block.text, context=context)
                output.append(Block(text=f"I'll remember: {input_block.text}"))
        return output


if __name__ == "__main__":
    tool = VectorSearchLearnerTool()
    repl = ToolREPL(tool)

    with repl.temporary_workspace() as client:
        repl.run_with_client(
            client, context=with_llm(context=AgentContext(), llm=OpenAI(client=client))
        )
