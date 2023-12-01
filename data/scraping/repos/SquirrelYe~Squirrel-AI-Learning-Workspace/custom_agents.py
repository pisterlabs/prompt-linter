from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish

search = SerpAPIWrapper()
searchTool = Tool(name="Search", func=search.run, description="useful for when you need to answer questions about current events", return_direct=True)
tools = [searchTool]

# 一个代理 (Agent) 由二个部分组成：
#   工具 tool：代理可以使用的工具。
#   代理执行器 ：这决定了采取哪些行动。


# 定义一个假的Agent
class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
 
        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.
 
        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
 
        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.
 
        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")


agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run("How many people live in canada as of 2023?")

# 返回结果
#   > Entering new AgentExecutor chain...
#   Canada's population was estimated at 39,858,480 on April 1, 2023, an increase of 292,232 people (+0.7%) from January 1, 2023.