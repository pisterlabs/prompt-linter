# ------- pdf_langchain virtualenv -----
# langchain
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain 
from langchain.prompts.prompt import PromptTemplate
import os 

# use python-dotenv to get API key
from dotenv import load_dotenv
load_dotenv()

os.environ.get("OPENAI_API_KEY")




# ---------SQL Chain from langchain example---------
db = SQLDatabase.from_uri("sqlite:///demo.db")

llm = OpenAI(temperature=0, verbose=True)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
#db_chain.run("How many people are there?")
db_chain.run("What are the names of the people?")