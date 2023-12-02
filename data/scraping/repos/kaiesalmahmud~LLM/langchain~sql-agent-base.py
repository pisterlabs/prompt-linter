import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

API_KEY = open('key.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

from dotenv import load_dotenv
load_dotenv()

host="localhost"
port="5432"
database="ReportDB"
user="postgres"
password="postgres"

db = SQLDatabase.from_uri(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

# Who are the top 5 retailers for the month of May?
# Summarize the weekly performance for me (Shops with 10 hours or more playtime, Shops with less than 10 hours, playtime Shops that has 0 playtime)
# What is the average of the shops that performed over 10 hours?

agent_executor.run("Describe the database")

