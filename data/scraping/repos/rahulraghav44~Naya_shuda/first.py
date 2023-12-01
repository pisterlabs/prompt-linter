from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.utilities.sql_database import SQLDatabase

from langchain_experimental.sql.base import SQLDatabaseChain




load_dotenv()

dburi ="sqlite:///data/.laon.db"
db=SQLDatabase.from_uri(dburi)
llm = OpenAI(temperature=0)
db_chain=SQLDatabaseChain(llm=llm,database=db,verbose=True)


db_chain.run("How many rows is in the responses rables of this db?")
db_chain.run("what are the top tree state where response comes are from?")
