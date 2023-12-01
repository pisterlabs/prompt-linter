import os
import sqlalchemy as db
from sqlalchemy import select
from sqlalchemy.sql import text
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

# OpenAI part
os.environ["OPENAI_API_KEY"] = "..."

# integratedML connection
db = SQLDatabase.from_uri("...")

llm = OpenAI(temperature=0)

# template to be adapted for our use case
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the employee table.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)

# query db through chatGPT
db_chain.run("How many passengers were transported?")
