from langchain.agents import AgentExecutor, Tool, AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.utilities import SerpAPIWrapper
import os

from real_tools.search_flight_ticket_tool import SearchFlightTool
from real_tools.search_hotel_tool import SearchHotelTool
from real_tools.search_sight_ticket_tool import SearchSightTicketTool
from real_tools.search_train_ticket_tool import SearchTrainTool

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "886ab329f3d0dda244f3544efeb257cc077d297bb0c666f5c76296d25c0b2279"

search = SerpAPIWrapper()

from langchain.schema import SystemMessage

system_message = SystemMessage(
    content="""You are the customer service of the travel counselor Little Camel, who can provide customers with travel plan recommendations and give specific schedules;
    You won't fabricate facts, and you will try your best to collect facts and data to support your researchPlease ensure that you complete the above objectives according to the following rules:
    1/ You should ask enough questions to determine the customer's travel preferences, such as domestic and foreign destinations, specific cities, travel time, travelers, whether there are children accompanying them (need to ask their age), travel style, transportation methods, and other information;
    2/ If there are relevant search results, you should provide them;
    3/After crawling and searching, you should think, "Should I search for information based on the user needs I have collected to get better planning?" If the answer is yes, continue;
    4/ You shouldn't make up things, you should only write the facts and data you collected;
    5/ In the final output, you should list the specific travel schedule for the customer;
    6/ 请使用中文回复."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

tools = [
    SearchFlightTool(),
    SearchHotelTool(),
    SearchSightTicketTool(),
    SearchTrainTool()
]

llm = ChatOpenAI(temperature=0.5, openai_api_key="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", model="gpt-3.5-turbo-1106")

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=300)

fakeAgent = initialize_agent(
    tools,  # 配置工具集
    llm,  # 配置大语言模型 负责决策
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,  # 设定 agent 角色
    memory=memory,  # 配置记忆模式
)


def random_word(query: str) -> str:
    print("\nNow I'm doing this!")
    return "foo"
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="RandomWord",
        func=random_word,
        description="call this to get a random word.",
    ),
]
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=fakeAgent, tools=tools, verbose=True
)
agent_executor.run("How many people live in canada as of 2023?")