#%%
## Libraries
from utils.ai import LLM
import vertexai

## Variables
PROJECT_ID = "vtxdemos"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

vertexai.init(project="vtxdemos", location="us-central1")

llm = LLM(
    bq_source="cloud-llm-preview4.sockcop_dataset.billing_latest", 
    text_model="text-bison@001",)

text_llm, embeddings = llm.LoadModels()
df = llm.LoadBQToPandas()
db_chain = llm.SqlToBigqueryEngine()

## ------------------------------------------------------------------------------------ ##

# %%
# @title Using GoogleSQL Prompt template to guide the LLM to generate output
## Initial Prompt
from langchain.prompts.prompt import PromptTemplate

_googlesql_prompt = """You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the following tables:
{table_info}

Question: {input}"""

GOOGLESQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_googlesql_prompt,
)

## Final Prompt
# %%
# Test 1
final_prompt = GOOGLESQL_PROMPT.format(input='What is the total consumption for AlloyDB on April?', table_info='user_activity', top_k=100)
db_chain.run(final_prompt)
# %%
# Test 2
final_prompt = GOOGLESQL_PROMPT.format(input='What was the total consumption until now for europe-west8?', table_info='user_activity', top_k=100)
db_chain.run(final_prompt)

# %%
