# SQL file
import os
from sqlalchemy import create_engine
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from tools.dbbase import SQLDatabase
from tools.dbchain import SQLDatabaseChain

with open("openai_api_key.txt", "r") as f:
    api_key = f.read()
    

os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    temperature=0.4,
)

# sql = create_engine(
#     "sqlite:///sayvai.db",
#     echo=True,
#     future=True,
# )

    
db = SQLDatabase.from_uri("sqlite:///sayvai.db")
sql_db_chain = SQLDatabaseChain.from_llm(
llm=llm,
db=db,
)

sql_db_chain.run("insert employee kedar, mobile - 9791642132")