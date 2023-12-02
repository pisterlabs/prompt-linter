from decouple import config
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from langchain.utilities.dalle_image_generator import DallEAPIWrapper


openai_api_key = config("OPENAI_API_KEY")
wiki = WikipediaAPIWrapper()
duck = DuckDuckGoSearchAPIWrapper(region="en-us", max_results=10)

tools = [
    Tool(
        name="wikipedia",
        func=wiki.run,
        description="Useful for when you need detailed information on a topic from wikipedia.",
    ),
    Tool(
        name="duckduckgo",
        func=duck.run,
        description="Useful for when you need to search the internet for something another tool cannot find.",
    ),
]


chat_gpt_api = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model="gpt-3.5-turbo-0613",
)

agent = initialize_agent(
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    llm=chat_gpt_api,
    tools=tools,
    verbose=True,
    max_iterations=10,
)


dalle = DallEAPIWrapper(openai_api_key=openai_api_key, n=1, size="512x512")


dalle_tool = Tool(
    name="dalle",
    func=dalle.run,
    description="Useful for when you need to generate images of something.",
)

tools.append(dalle_tool)


agent_v2 = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=chat_gpt_api,
    tools=tools,
    verbose=True,
    max_iterations=10,
)


agent_v2.run("I would like an image of a flying spaghetti monster.")
