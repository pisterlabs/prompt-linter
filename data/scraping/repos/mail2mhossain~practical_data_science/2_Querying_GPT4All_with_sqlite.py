import os
from dotenv import load_dotenv

from langchain import SQLDatabase, SQLDatabaseChain
from langchain import PromptTemplate
from langchain.llms import GPT4All
from langchain import HuggingFaceHub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv("../.env")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_name = "MBZUAI/LaMini-Flan-T5-783M"  # LaMini-Flan-T5-783M  LaMini-GPT-1.5B

local_path = (
    "Z:/MHossain_OneDrive/OneDrive/ChatGPT/LLM/Models/ggml-gpt4all-j-v1.3-groovy.bin"
)
# callbacks = [StreamingStdOutCallbackHandler()]
# llm = GPT4All(model=local_path, callbacks=callbacks, backend="gptj", verbose=True)
llm = HuggingFaceHub(
    repo_id=model_name,
    model_kwargs={"temperature": 0, "max_length": 512},
)

db = SQLDatabase.from_uri(
    "sqlite:///Chinook_Sqlite.sqlite", include_tables=["Employee"]
)

# Define the prompt template
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

Question: {input}"""

# Create db chain
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info"], template=_DEFAULT_TEMPLATE
)

query = "How many employees are there?"
# query = "How many albums by Aerosmith?"
# query = "How many employees are also customers?"

# Create an instance of SQLDatabaseChain
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)

response = db_chain(query)
print(response)
