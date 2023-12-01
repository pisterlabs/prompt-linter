import os
import openai
from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

load_dotenv()


# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a completion - qna를 위하여 davinci 모델생성
llm = AzureOpenAI(deployment_name="text-davinci-003")

# dburi = os.getenv("DATABASE_URL")
dburi = "sqlite:///db/chinook.db"
db = SQLDatabase.from_uri(dburi)
# llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

db_chain.run("How many rows is in the responses table of this db?")
db_chain.run("Describe the responses table")
db_chain.run("What are the top 3 countries where these responses are from?")
db_chain.run("Give me a summary of how these customers come to hear about us. \
    What is the most common way they hear about us?")
