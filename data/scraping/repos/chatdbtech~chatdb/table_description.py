import openai
from uuid import uuid4

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
import psycopg2
from psycopg2 import sql
import json
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from llm import llm, chat_llm, embeddings

table_name = 'rajasthan_dlc_rate_final'
column_name = 'residential'
db_uri = 'postgresql://apoorvagarwal@localhost:5432/parceldb'



def get_unique_values(table_name, column_name, db_uri):
    distinct_query = "select distinct " + column_name + " from " + table_name
    connection = psycopg2.connect(db_uri)
    cursor = connection.cursor()
    cursor.execute(distinct_query)
    values = cursor.fetchall()


    


def get_top_10_unique_values(cursor, table_name, column_name):
    print()
    distinct_query = 'select distinct "' + column_name + '" from "' + table_name + '" limit 10'
    # connection = psycopg2.connect(db_uri)
    # cursor = connection.cursor()
    cursor.execute(distinct_query)
    values = cursor.fetchall()
    print(values)
    if len(values) == 0:
        return f"There are no values in this column."
    elif len(values) < 10:
        return f"These are the only values in this column: {str(values)}"
    else:
        return f"Some of the values of this column are {str(values)}"



# def get_top_100_unique_values(table_name, column_name, db_uri):
#     distinct_query = "select distinct " + column_name + " from " + table_name + " limit 100"
#     connection = psycopg2.connect(db_uri)
#     cursor = connection.cursor()
#     cursor.execute(distinct_query)
#     values = cursor.fetchall()




def write_description_of_column(csv_file, table_name, column_name):
    df = pd.read_csv(csv_file)
    df_table = df[df['table_name'] == table_name]
    result = "\n".join([f"{index}:\n{row.to_dict()}" for index, row in df_table.iterrows()])
    

    template = "Here is some information about an SQL table.:\n " + result + "\n" + "Describe what the {column_name} could mean."
    prompt = PromptTemplate(template=template, input_variables=["column_name"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_chain.predict(column_name=column_name)





