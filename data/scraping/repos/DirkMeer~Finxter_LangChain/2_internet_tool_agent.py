from decouple import config
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from tools import InternetTool


chat_gpt_api = ChatOpenAI(
    model="gpt-3.5-turbo-0613",
    openai_api_key=config("OPENAI_API_KEY"),
    temperature=0,
)

agent = initialize_agent(
    llm=chat_gpt_api,
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=[InternetTool()],
    verbose=True,
)

agent.run("16 Pineapples -3 Pineapples = ?")
