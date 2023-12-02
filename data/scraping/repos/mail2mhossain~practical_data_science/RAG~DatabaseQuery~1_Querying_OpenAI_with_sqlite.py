import os
from dotenv import load_dotenv

import psycopg2
from langchain import SQLDatabase, SQLDatabaseChain, OpenAI
from langchain.chat_models import ChatOpenAI

load_dotenv("../.env")

OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")


# PostgreSQL
# Setup database
# db = SQLDatabase.from_uri(
#     f"postgresql+psycopg2://postgres:{os.getenv('DBPASS')}@localhost:5432/{os.getenv('DATABASE')}",
# )


# Sqlite
db = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")

# llm = OpenAI(
#     openai_api_key=OpenAI_API_KEY, temperature=0, verbose=True
# )  # model_name="gpt-3.5-turbo",
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OpenAI_API_KEY
)

# query = "How many employees are there?"
# query = "How many albums by Aerosmith?"
query = "How many employees are also customers?"

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False)  # use_query_checker=

# Return Intermediate Steps
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=False, use_query_checker=True, return_intermediate_steps=True)

result = db_chain.run(query)


print(result)
# result["intermediate_steps"]
