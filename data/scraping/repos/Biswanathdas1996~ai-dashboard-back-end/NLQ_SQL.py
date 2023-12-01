import os
from langchain import SQLDatabase
from langchain.llms import AzureOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
import pandas as pd
from sqlalchemy import create_engine, inspect,  MetaData, Table
import json
from decimal import Decimal
from datetime import date


_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"\n
SQLQuery: "SQL Query to run"\n
SQLResult: "Result of the SQLQuery"\n
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the employee table.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)


def connect(dbUri):
    input_db = SQLDatabase.from_uri(dbUri)
    return input_db


def getAllTables(dbUri):
    # postgresql://Biswanathdas:Papun$1996@post-db-ai.postgres.database.azure.com/azure-sales-data
    engine = create_engine(dbUri)
    print("========================>", dbUri)
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    print(table_names)
    return table_names


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, date):
            return obj.isoformat()  # Convert date to string in ISO format
        return super(CustomEncoder, self).default(obj)


def fetchAllDataFromTable(dbUri, table_name):
    engine = create_engine(dbUri)
    # Connect to the database
    connection = engine.connect()

    metadata = MetaData()
    metadata.bind = engine  # Bind the engine to the metadata

    table = Table(table_name, metadata, autoload_with=engine)

    # Query all data from the table
    results = connection.execute(table.select()).fetchall()

    # Get column names
    column_names = table.columns.keys()

    # Convert data to JSON format
    data = [dict(zip(column_names, row)) for row in results]

    json_data = json.dumps(data, indent=4, cls=CustomEncoder)
    print("json_data", json_data)
    # Close the connection
    connection.close()
    return json_data


def searchInDB(config, dbUri, question):

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = config['OPENAI_API_VERSION']
    os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']
    os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
    print(config)
    db = connect(dbUri)
    llm_1 = AzureOpenAI(deployment_name=config['deployment_name'],
                        model_name="gpt-35-turbo",)

    db_agent = SQLDatabaseChain(llm=llm_1,
                                database=db,
                                verbose=False,
                                # prompt=PROMPT
                                )

    result = db_agent.run(question)
    return result
