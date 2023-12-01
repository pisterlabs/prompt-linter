from langchain import OpenAI, SQLDatabase
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.llms.openai import OpenAI 
from langchain.agents import AgentExecutor 
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import sqlalchemy
from sqlalchemy import create_engine, text, exc
from sqlalchemy.dialects.mssql import pymssql
from sqlalchemy import text
import os.path
engine = create_engine("sqlite:///chinook.db")
db = SQLDatabase.from_uri("sqlite:///chinook.db")
OPENAI_API_KEY = "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"
gpt = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')
toolkit = SQLDatabaseToolkit(db=db, llm=gpt)
agent_executor = create_sql_agent(
    llm=gpt,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
question = "Who are the top 3 best selling artists?"
agent_executor.run(question)
