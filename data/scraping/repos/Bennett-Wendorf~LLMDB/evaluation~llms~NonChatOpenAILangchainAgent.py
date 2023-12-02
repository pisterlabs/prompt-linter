#region Imports
# # Environment variables
from dotenv import load_dotenv
load_dotenv()

# Langchain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

# Base
from llms.BaseLLM import BaseLLM
#endregion

class NonChatOpenAILangchainAgent(BaseLLM):
    def __init__(self):
        self.llm = OpenAI(temperature=0)

        self.db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

        self.agent_executor = create_sql_agent(
            llm = self.llm,
            toolkit = self.toolkit,
            verbose = False,
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    def inference(self, query: str) -> str:
        return self.agent_executor.run(query)