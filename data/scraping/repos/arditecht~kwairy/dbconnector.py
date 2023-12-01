############# Production Settings example for MySQL DB #############
'''
db_user = "root"
db_password = "pass1234" #Enter you password database password here
db_host = "localhost"  
db_name = "test_db" #name of the database
db_port = "0000" #specify your port here
connection_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
'''

############# Development Settings for SQLite DB #############
# NOTE: Using local sqlite server for dev and testing purposes


import os
import openai
from llama_index import SQLDatabase,ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from IPython.display import Markdown, display
from llama_index.logger import LlamaLogger

from sqlalchemy import select, create_engine, MetaData, Table, inspect, String, Integer, column


class DBcomm :
    # working on a sample database for now
    connection_uri = "sqlite:///dev/chinook.db" # "your DB connection uri like in the example above" --For now using sqlite for dev and testing purposes
    sql_engine = create_engine(connection_uri)