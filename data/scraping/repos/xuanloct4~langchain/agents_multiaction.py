
import environment
from agents_tools import search_tool_serpapi

from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent

def random_word(query: str) -> str:
    print("\nNow I'm doing this!")
    return "foo"

tools = [
    search_tool_serpapi(),
    Tool(
        name = "RandomWord",
        func=random_word,
        description="call this to get a random word."
    
    )
]
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish

class FakeAgent(BaseMultiActionAgent):
    """Fake Custom Agent."""
    
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool="Search", tool_input=kwargs["input"], log=""),
                AgentAction(tool="RandomWord", tool_input=kwargs["input"], log=""),
            ]
        else:
            return AgentFinish(return_values={"output": "bar"}, log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool="Search", tool_input=kwargs["input"], log=""),
                AgentAction(tool="RandomWord", tool_input=kwargs["input"], log=""),
            ]
        else:
            return AgentFinish(return_values={"output": "bar"}, log="")
agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
print(agent_executor.run("How many people live in canada as of 2023?"))