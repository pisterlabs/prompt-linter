from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import os
from langchain import HuggingFaceHub, PromptTemplate

os.environ["HUGGINGFACEHUB_API_TOKEN"] = 'paste your token here'

db = SQLDatabase.from_uri("postgresql://postgres:docker@localhost:5432/postgres")

llm = OpenAI(openai_api_key='paste your token here', temperature=0)

repo_id = 'bigcode/starcoder'
# llm = HuggingFaceHub(
#     repo_id=repo_id, model_kwargs={"temperature": 0.5,"max_length": 64}
# )


_DEFAULT_TEMPLATE = """first create a syntactically correct {dialect} query,
query should not have anything other than sql query terminated by semicolon. 
Use the following format:
 
Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Answer to the question

Only use the following tables:
{table_info}
  
Use the table authors with author_id as primary key to get the list of authors and the table blogs with blog_id as the 
primary key to get the list of blogs and join the two tables on author_id. 
Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)


#db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=PROMPT, verbose=True)

db_chain.run("what is name of student who has minimum age?")
# db_chain.run("total number of students?")
# db_chain.run("list students starting with letter 'a'?")