from llama_hub.youtube_transcript.base import YoutubeTranscriptReader
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import ServiceContext, LLMPredictor, VectorStoreIndex
from llama_index.storage import StorageContext
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from llama_index.query_engine import SQLAutoVectorQueryEngine
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, column
from llama_index.indices.struct_store.sql import SQLStructStoreIndex
from llama_index.langchain_helpers.sql_wrapper import SQLDatabase
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
import gradio as gr
import os, sys
import logging

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#create chroma client and collection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("national-parks-things-to-do")

#define node parser and LLM
chunk_size = 1024
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True))
service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm_predictor=llm_predictor)

text_splitter = TokenTextSplitter(chunk_size=chunk_size)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

#create empty vector index (to be filled in later)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)

#create SQL engine
engine = create_engine("sqlite:///national_parks.db", future=True)
metadata_obj = MetaData()
metadata_obj.drop_all(engine)

#create national_parks table
table_name = "national_parks"
national_parks_table = Table(
    table_name,
    metadata_obj,
    Column("park_name", String(50), primary_key=True),
    Column("average_june_temperature", String(20)),
    Column("elevation_highest_point", String(20))
)
metadata_obj.create_all(engine)

#print tables
metadata_obj.tables.keys()

#insert data
from sqlalchemy import insert
rows = [
    {"park_name": "Glacier National Park", "average_june_temperature": "60-70", "elevation_highest_point": "10,000"},
    {"park_name": "Yellowstone National Park", "average_june_temperature": "60-75", "elevation_highest_point": "11,000"},
    {"park_name": "Rocky Mountain National Park", "average_june_temperature": "60-75", "elevation_highest_point": "14,000"},
]
for row in rows:
    stmt = insert(national_parks_table).values(**row)
    with engine.connect() as connection:
        cursor = connection.execute(stmt)
        connection.commit()

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM national_parks")
    print(cursor.fetchall())
    

def data_ingestion_indexing():

    #loads data on things to do in those 3 national parks from YouTube videos
    youtube_loader = YoutubeTranscriptReader()
    youtube_documents = youtube_loader.load_data(ytlinks=['https://www.youtube.com/watch?v=poBfOPFGgUU',
                                                          'https://www.youtube.com/watch?v=KLWv8TsKcGc',
                                                          'https://www.youtube.com/watch?v=UV4tENBS0mQ'])    

    #build sql index
    sql_database = SQLDatabase(engine, include_tables=["national_parks"])
    sql_index = SQLStructStoreIndex.from_documents(
        [], 
        sql_database=sql_database, 
        table_name="national_parks",
    )

    #insert documents into vector index. Each document has metadata of the park attached
    parks = ['Glacier National Park', 'Yellowstone National Park', 'Rocky Mountain National Park']
    for park, youtube_document in zip(parks, youtube_documents):
        nodes = node_parser.get_nodes_from_documents([youtube_document])
        # add metadata to each node
        for node in nodes:
            node.extra_info = {"title": park}
        vector_index.insert_nodes(nodes)

    #create SQL query engine
    sql_query_engine = sql_index.as_query_engine(synthesize_response=True)

    #create vector query engine
    vector_store_info = VectorStoreInfo(
        content_info='things to do in different national parks',
        metadata_info=[
            MetadataInfo(
                name='title', 
                type='str', 
                description='The name of the national park'),
        ]
    )
    vector_auto_retriever = VectorIndexAutoRetriever(vector_index, vector_store_info=vector_store_info)
    retriever_query_engine = RetrieverQueryEngine.from_args(
        vector_auto_retriever, service_context=service_context
    )
    
    #create SQL query tool
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            'Useful for translating a natural language query into a SQL query over a table containing: '
            'national_parks, containing the average_june_temperature/elevation_highest_point of each national park'
        )
    )
    
    #create vector query tool
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=retriever_query_engine,
        description=f'Useful for answering semantic questions about different national parks',
    )

    #define SQLAutoVectorQueryEngine
    query_engine = SQLAutoVectorQueryEngine(
        sql_tool,
        vector_tool, 
        service_context=service_context
    )

    return query_engine

def data_querying(input_text):

    #queries the engine with the input text    
    response = query_engine.query(input_text)
    return response.response
    

iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=3, label="Enter your question"),
                     outputs="text",
                     title="Things to do in National Parks")

#data ingestion and indexing
query_engine = data_ingestion_indexing()
iface.launch(share=False)
