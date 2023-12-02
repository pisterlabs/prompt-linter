import sqlite3

import langchain
import pandas as pd
import st_pages
import streamlit as st
from langchain import OpenAI, PromptTemplate, SQLDatabase
from langchain.callbacks import get_openai_callback
from langchain_experimental.sql import SQLDatabaseChain
from openbb_terminal.sdk import openbb
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

langchain.debug = True

st_pages.add_page_title()  # type: ignore

st.title("Dashboard powered by OpenBB :butterfly: & ChatGPT :robot_face:")
ticker = st.selectbox(
    label="Choose a dataset to load",
    options=(
        "anes96",
        "cancer",
        "ccard",
        "cancer_china",
        "co2",
        "committee",
        "copper",
        "cpunish",
        "danish_data",
        "elnino",
        "engel",
        "fair",
        "fertility",
        "grunfeld",
        "heart",
        "interest_inflation",
        "longley",
        "macrodata",
        "modechoice",
        "nile",
        "randhie",
        "scotland",
        "spector",
        "stackloss",
        "star98",
        "statecrim",
        "strikes",
        "sunspots",
        "wage_panel",
    ),
)

uploaded_file = st.file_uploader("...Or load your own custom CSV dataset")

table_name = "statesdb"
uri = "file:memory?cache=shared&mode=memory"
openai_key = st.secrets["OPENAI_KEY"]


@st.cache_data()
def load(ticker: str | None):
    df = openbb.econometrics.load(ticker)
    return df


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # type: ignore
    st.write(df)  # type: ignore
else:
    df = load(ticker)
    st.write(df)  # type: ignore

query = st.text_input(
    label="Any questions?", help="Ask any question based on the loaded dataset"
)
conn = sqlite3.connect(uri, uri=True)
df.to_sql(table_name, conn, if_exists="replace", index=False)  # type: ignore

db_eng = create_engine(
    url="sqlite:///file:memdb1?mode=memory&cache=shared",
    poolclass=StaticPool,
    creator=lambda: conn,
)
db = SQLDatabase(engine=db_eng)

template = """You are an SQLite expert. Given an input question, first create a
syntactically correct SQLite query to run, then look at the results of the query and
return the answer to the input question. Unless the user specifies in the question a
specific number of examples to obtain, query for at most {top_k} results using the LIMIT
clause as per SQLite. You can order the results to return the most informative data in
the database. Never query for all columns from a table. You must query only the columns
that are needed to answer the question. Wrap each column name in double quotes (") to
denote them as delimited identifiers. Pay attention to use only the column names you
can see in the tables below. Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}
"""
prompt = PromptTemplate(
    input_variables=["input", "table_info", "top_k"], template=template
)
lang_model = OpenAI(
    openai_api_key=openai_key,
    temperature=0,
    max_tokens=300,
    client=None,
)
db_chain = SQLDatabaseChain.from_llm(  # type: ignore
    llm=lang_model,
    db=db,
    prompt=prompt,
    use_query_checker=False,
)

if query:
    with get_openai_callback() as cb:
        response = db_chain.run(query)
        st.sidebar.write(  # type: ignore
            f"Your request costs: {str(cb.total_cost)} USD"
        )
    st.write(response)  # type: ignore
