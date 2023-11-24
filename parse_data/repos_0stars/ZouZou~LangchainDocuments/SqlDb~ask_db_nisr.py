import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import pymssql
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

database_user = os.getenv("DATABASE_USERNAME")
database_password = os.getenv("DATABASE_PASSWORD")
database_server = os.getenv("DATABASE_SERVER")
database_db = os.getenv("DATABASE_DB")

connection_string = f"mssql+pymssql://{database_user}:{database_password}@{database_server}/{database_db}"

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table policy or policies, they really mean the polcom table.
Do not use LIMIT statements and always get the top 3 results

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

custom_table_info = {
    "POLCOM": """CREATE TABLE POLCOM (
	[rec_type] [varchar](2) NOT NULL,
	[product] [varchar](6) NOT NULL,
	[policy_no] [int] NOT NULL,
    [pol_serno] [int] NOT NULL,
	[pin] [int] NOT NULL,
	[total_premium] [decimal](18, 3) NULL,
	PRIMARY KEY ("pol_serno")
)
""",
    "PERSONS": """CREATE TABLE PERSONS (
	pin INTEGER NOT NULL IDENTITY(1,1),
	firstname NVARCHAR(120) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL,
	father NVARCHAR(40) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
    family NVARCHAR(60) COLLATE SQL_Latin1_General_CP1_CI_AS NOT NULL,
	PRIMARY KEY ("pin")
)
"""
}

db = SQLDatabase.from_uri(
    connection_string,
    include_tables=['PERSONS', 'POLCOM'], # we include only one table to save tokens in the prompt :)
    sample_rows_in_table_info=2,
    custom_table_info=custom_table_info
    )
# toolkit = SQLDatabaseToolkit(db=db, llm=llm, reduce_k_below_max_tokens=True)
# agent_executor = create_sql_agent(
#     llm=llm,
#     toolkit=toolkit,
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
# )




db_chain = SQLDatabaseChain.from_llm(
    llm, 
    db, 
    prompt=PROMPT, 
    verbose=True, 
    use_query_checker=True, 
    return_intermediate_steps=True, 
    top_k=3)
result = db_chain("which person has the most policies?")
print(result["intermediate_steps"])

# agent_executor.run("Which person has the most polcoms with rec_type equal to 1?")
