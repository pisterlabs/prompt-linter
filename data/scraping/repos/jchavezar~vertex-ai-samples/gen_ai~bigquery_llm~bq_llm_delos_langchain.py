#%%
#region Libraries
import pandas as pd
import streamlit as st
from sqlalchemy import *
from sqlalchemy.schema import *
from langchain.llms import VertexAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy.engine import create_engine
from langchain.prompts.prompt import PromptTemplate
#endregion

#region Variables
project_id="vtxdemos"
region="us-central1"
model_name="text-bison"
dataset_name="custom"
table_name="delo"
#endregion

#region Vertex AI LLM
llm = VertexAI(
    project_id=project_id,
    region=region,
    model_name=model_name,
)
engine = create_engine(f'bigquery://{project_id}/custom')
engine.execute(f'SELECT * FROM `{project_id}.{dataset_name}.{table_name}`').first()
db = SQLDatabase(engine=engine,metadata=MetaData(bind=engine),include_tables=[table_name])
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
#endregion

#region Query
_googlesql_prompt = """You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run. This will be the SQLQuery. Then look at the results of the query. This will be SQLResults. Finally return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
When running SQLQuery across BigQuery you must only include BigQuery SQL code from SQLQuery.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Only use the following tables:
{table_info}

Question: {input}"""

GOOGLESQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_googlesql_prompt,
)
prompt="What was the partner with the best EAC revenue?"
#endregion

#prompt=st.text_input(label="Prompt:")
final_prompt = GOOGLESQL_PROMPT.format(input=prompt, table_info='custom.delo', top_k=100)
x=db_chain.run(final_prompt)

#st.write()
# %%

from google.cloud import bigquery

client = bigquery.Client()
df=client.query("select partner, sum(EAC_revenue) as total_EAC_revenue from `vtxdemos.custom.delo` group by partner order by total_EAC_revenue DESC").to_dataframe()
df.head()
# %%
