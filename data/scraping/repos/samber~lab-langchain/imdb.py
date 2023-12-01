
import os

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

API_KEY = os.environ.get('OPENAI_API_KEY')

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://screeb:screeb@51.159.148.147:5432/screeb",
    # f"postgresql+psycopg2://screeb:screeb@localhost:5432/screeb",
)

# llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name='gpt-4')
llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name='gpt-3.5-turbo-16k')

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

{question}
"""

def ask(prompt):
    question = QUERY.format(question=prompt)
    return db_chain.run(question)

def get_prompt():
    print("Type 'exit' to quit")

    while True:
        prompt = input("Enter a prompt: ")

        if prompt.lower() == 'exit':
            print('Exiting...')
            break
        else:
            try:
                print(ask(prompt))
            except Exception as e:
                print(e)

if __name__ == "__main__":
    get_prompt()
