from langchain.agents import AgentExecutor, BaseMultiActionAgent
import os

from real_tools.search_flight_ticket_tool import SearchFlightTool
from real_tools.search_hotel_tool import SearchHotelTool
from real_tools.search_sight_ticket_tool import SearchSightTicketTool
from real_tools.search_train_ticket_tool import SearchTrainTool

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "886ab329f3d0dda244f3544efeb257cc077d297bb0c666f5c76296d25c0b2279"


tools = [
    SearchFlightTool(),
    SearchHotelTool(),
    SearchSightTicketTool(),
    SearchTrainTool()
]
from typing import Any, List, Tuple, Union

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
                AgentAction(tool="SearchFlightTool", tool_input=kwargs["input"], log=""),
                AgentAction(tool="SearchTrainTool", tool_input=kwargs["input"], log=""),
                AgentAction(tool="SearchHotelTool", tool_input=kwargs["input"], log=""),
                AgentAction(tool="SearchSightTicketTool", tool_input=kwargs["input"], log=""),
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
                AgentAction(tool="search_flight", tool_input=kwargs["input"], log=""),
                AgentAction(tool="search_hotel", tool_input=kwargs["input"], log=""),
                AgentAction(tool="search_sight_ticket", tool_input=kwargs["input"], log=""),
                AgentAction(tool="search_train", tool_input=kwargs["input"], log=""),
            ]
        else:
            return AgentFinish(return_values={"output": "bar"}, log="")



llm=OpenAI(temperature=0)
# agent = FakeAgent(llm= ,tools=tools)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
agent_executor.run("""You are the customer service of the travel counselor Little Camel, who can provide customers with travel plan recommendations and give specific schedules;
    You won't fabricate facts, and you will try your best to collect facts and data to support your researchPlease ensure that you complete the above objectives according to the following rules:
    1/ You should ask enough questions to determine the customer's travel preferences, such as domestic and foreign destinations, specific cities, travel time, travelers, whether there are children accompanying them (need to ask their age), travel style, transportation methods, and other information;
    2/ If there are relevant search results, you should provide them;
    3/After crawling and searching, you should think, "Should I search for information based on the user needs I have collected to get better planning?" If the answer is yes, continue;
    4/ You shouldn't make up things, you should only write the facts and data you collected;
    5/ In the final output, you should list the specific travel schedule for the customer;
    6/ 请使用中文回复."""
                   + "北京三日游?")