from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain.chains import SQLDatabaseSequentialChain


db = SQLDatabase.from_uri("sqlite:///KnowledgeBase/list_products.db")
llm = OpenAI(temperature=0, verbose=True)
db_chain = SQLDatabaseSequentialChain.from_llm(llm, db, verbose=True)
