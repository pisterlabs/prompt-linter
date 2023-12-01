import streamlit as st

import pandas as pd

from cqlsession import getCQLKeyspace, getCQLSession

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Cassandra
from langchain.document_loaders import DataFrameLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator

from langchain.llms import OpenAI

from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv(), override=True)
ASTRA_DB_KEYSPACE = os.environ["ASTRA_DB_KEYSPACE"]

#Globals
cqlMode = 'astra_db'
table_name = 'vs_rca_openai'
llm = OpenAI()
embedding = OpenAIEmbeddings()

session = getCQLSession(mode=cqlMode)

cassandra_vstore = Cassandra(
    embedding=embedding,
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name=table_name,
)

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Cassandra,
    embedding=embedding,
    vectorstore_kwargs={
        'session': session,
        'keyspace': ASTRA_DB_KEYSPACE,
        'table_name': table_name,
    },
)

"""load_support_tickets

Load the support tickets file into a dataframe 
"""
def load_support_tickets():
    df = pd.read_csv("./data/rca/customer_support_tickets.csv")

    # fixing the ticket description product purchased macro
    df["Ticket Description"]= df.apply(lambda x: x['Ticket Description'].replace('{product_purchased}', x['Product Purchased']), axis=1)
    df['embedding'] = df['Ticket Subject'] + " " + df['Ticket Description']

    # Removing tickets without a RCA
    mask = df['Time to Resolution'].notnull()
    new_df = df[mask]    

    loader = DataFrameLoader(new_df, page_content_column="embedding")
    docs = loader.load()

    cassandra_vstore.clear()
    with st.spinner('Loading support ticket docs...'):
        index_creator.from_documents(docs)


"""sim_search

Returns a list of k documents where k=3
"""
def sim_search(text_embedding):
    return cassandra_vstore.search(text_embedding, search_type='similarity', k=3)