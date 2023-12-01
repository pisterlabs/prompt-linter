import os

from typing import List, Tuple, Any, Union

from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
from langchain.schema import AgentAction, AgentFinish

import config as c


os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY

search = SerpAPIWrapper()
tools = [
    Tool(
        name='Search',
        func=search.run,
        description='useful for when you need to answer questions about current events',
        return_direct=True,
    ),
]


class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ['input']

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool='Search', tool_input=kwargs['input'], log='')

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool='Search', tool_input=kwargs['input'], log='')


if __name__ == '__main__':
    agent = FakeAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    agent_executor.run('How old is Robert Downey Jr.?')
