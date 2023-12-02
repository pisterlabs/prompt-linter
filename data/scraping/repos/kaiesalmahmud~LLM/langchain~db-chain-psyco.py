import os
from langchain.agents import *
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.chains import SQLDatabaseSequentialChain
from langchain.prompts.prompt import PromptTemplate

import psycopg2
import pandas as pd
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

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

llm = ChatOpenAI(model_name="gpt-3.5-turbo")


_postgres_prompt = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

"""

PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}"""

POSTGRES_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_postgres_prompt + PROMPT_SUFFIX,
)


# Setup the database chain
db_chain = SQLDatabaseSequentialChain.from_llm(llm=llm, database=db, query_prompt=POSTGRES_PROMPT, verbose=False, return_intermediate_steps=True)

# def get_prompt():
#     print("Type 'exit' to quit")

#     while True:
#         question = input("Enter your question: ")

#         if question.lower() == 'exit':
#             print('Exiting...')
#             break
#         else:
#             try:
#                 # question = QUERY.format(question=prompt)
#                 print(db_chain.run(question))
#             except Exception as e:
#                 print(e)

# get_prompt()

# result = db_chain("What the total play time for the month of May?")
result = db_chain("list the top 5 retailers")

print('SQL Query: ')
sql_query = result['intermediate_steps'][1]
print(sql_query)
print()

print('SQL Result: ')
print(result['intermediate_steps'][3])

print()
print('Answer: ')
print(result['intermediate_steps'][5])

# Define function to execute a query on the database and return the results as a dataframe
def execute_query(query, connection):
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(results, columns=columns)
    return df

# Define function to connect to the database using input parameters
def connect_to_database(host, port, username, password, database):
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=username,
        password=password
    )
    return conn

connection = connect_to_database(host, port, user, password, database)

result = execute_query(sql_query, connection)

print()
print(result)