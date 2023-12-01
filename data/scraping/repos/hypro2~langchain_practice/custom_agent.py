from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish

"""
자신 만의 사용자 정의 에이전트를 만드는 방법을 거칩니다.
에이전트는 두 부분으로 구성됩니다:

- 도구: 에이전트가 사용할 수 있는 도구입니다.
- 에이전트 클래스 자체: 어떤 작업을 수행할지 결정합니다.

"""

search = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True,
    )
]



class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """
        입력이 주어지면, 무엇을 해야 할지 결정합니다.

        Args:
            intermediate_steps: LLM이 현재까지 수행한 단계,
            **kwargs: 사용자 입력.

        Returns:
            사용할 도구를 지정하는 액션.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")



    # 비동기식 실행
    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """
        입력이 주어지면, 무엇을 해야 할지 결정합니다.

        Args:
            intermediate_steps: LLM이 현재까지 수행한 단계,
            **kwargs: 사용자 입력.

        Returns:
            사용할 도구를 지정하는 액션.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")


if __name__=="__main__":

    agent = FakeAgent()
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        verbose=True
                                                        )

    agent_executor.run("How many people live in canada as of 2023?")