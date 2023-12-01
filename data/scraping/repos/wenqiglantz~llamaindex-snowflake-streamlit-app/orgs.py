import streamlit as st
import pandas as pd
import logging, sys, os, utils as u
import json

# workaround for https://github.com/snowflakedb/snowflake-sqlalchemy/issues/380.
try:
    u.snowflake_sqlalchemy_20_monkey_patches()
except Exception as e:
    raise ValueError("Please run `pip install snowflake-sqlalchemy`")

import openai
from dotenv import load_dotenv
from snowflake.snowpark import Session, DataFrame
from llama_index import SQLDatabase, ServiceContext, LLMPredictor
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# connect to Snowflake
with open('credentials.json') as f:
    connection_parameters = json.load(f)  
session = Session.builder.configs(connection_parameters).create()

# function to load data
def data_loading(file) -> DataFrame:
    file_df = pd.read_csv(file)
    snowparkDf = session.write_pandas(file_df,table_name='ORGANIZATIONS',database='ORGANIZATIONS',schema='PUBLIC',quote_identifiers=True,auto_create_table=True, overwrite=True)
    return snowparkDf    

# function to query data
def data_querying():
    snowflake_uri = "snowflake://<username>:<password>@<org-account>/<database_name>/<schema_name>?warehouse=<warehouse_name>&role=<role_name>"
    
    #define node parser and LLM
    chunk_size = 1024
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm_predictor=llm_predictor)

    engine = create_engine(snowflake_uri)

    sql_database = SQLDatabase(engine)

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["ORGANIZATIONS"],
        service_context=service_context
    )

    question = st.text_area("Enter your question here")
    if question:
        response = query_engine.query(question)
        sql_query = response.metadata["sql_query"]
        print(sql_query)

        con = engine.connect()
        df = pd.read_sql(sql_query, con)
        st.write(df)
        st.bar_chart(df, x="Name", y="Number of employees")

st.header("Organization Data Loading and Querying")
file = st.file_uploader("Upload your CSV file here", type={"csv"})
if file is not None:
    df = data_loading(file)
    if df is not None:
        data_querying()

