import sqlite3
import pandas as pd
import llama_index
import os
import openai
from IPython.display import Markdown, display
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    text
)
from llama_index import SQLDatabase, ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine

# Create a new SQLite database (or connect to an existing one)
def create_and_load_db():
    # Connect to the SQLite database (or create a new one)
    conn = sqlite3.connect('company_info.db')

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv('C:\\Users\\Nsahni\\Downloads\\Github\\Bynd\\company_information_db.csv')

    # Write the data to a SQLite table
    df.to_sql('company_table', conn, if_exists='replace', index=False)
    return conn

def execute_query(conn, query):
    # Query the table
    query_result = pd.read_sql_query(query, conn)
    print(query_result)

conn = create_and_load_db()
with open('config.txt', 'r') as f:
    openai.api_key = f.read().strip()

llm = OpenAI(temperature=0, model="gpt-4")
engine = create_engine('sqlite:///company_info.db')
metadata_obj = MetaData()
metadata_obj.create_all(engine)
service_context = ServiceContext.from_defaults(llm=llm)
sql_database = SQLDatabase(engine, include_tables=['company_table'])
metadata_obj = MetaData()

# with engine.connect() as con:
#     rows = con.execute(text("SELECT * FROM company_table where market_cap > 10000000"))
#     for row in rows:
#         print(row)


query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["company_table"],
)
query_str = input("Please enter the query you are looking for: ")
response = query_engine.query(query_str)
# response_df = pd.DataFrame(response)
print(response)
# print(response)

# execute_query(conn, "SELECT * FROM company_table limit 10")
# Close the connection
conn.close()