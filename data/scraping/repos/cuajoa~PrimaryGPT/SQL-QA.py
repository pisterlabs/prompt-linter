# Script que se conecta a usa base de datos y responde preguntas en lenguaje natural

from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.sql_database import SQLDatabase
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print(_parentdir)

from scripts.config import Config

cfg = Config()

connection_string = f"mssql+pymssql://{cfg.database_username}:{cfg.database_password}@{cfg.database_server}:1436/{cfg.database_db}"

# mssql://[Server_Name[:Portno]]/[Database_Instance_Name]/[Database_Name]?FailoverPartner=[Partner_Server_Name]&InboundId=[Inbound_ID]  


db = SQLDatabase.from_uri(connection_string)
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)

# Create db chain
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{question}
"""


def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == "exit":
            print("Exiting...")
            break
        else:
            try:
                question = QUERY.format(question=prompt)
                print(db_chain.run(question))
            except Exception as e:
                print(e)


get_prompt()
