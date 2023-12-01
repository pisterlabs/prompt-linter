import openai
import os
import requests
import numpy as np
import pandas as pd
from typing import Iterator
import tiktoken
import textract
from numpy import array, average

from database import get_redis_connection, get_redis_results
from transformers import handle_file_string

# Set our default models and chunking size
from config import COMPLETIONS_MODEL, EMBEDDINGS_MODEL, CHAT_MODEL, TEXT_EMBEDDING_CHUNK_SIZE, VECTOR_FIELD_NAME, PREFIX, INDEX_NAME


##Additional Line
import config
openai.api_key = config.DevelopmentConfig.OPENAI_KEY


# Setup Redis
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

redis_client = get_redis_connection()
VECTOR_DIM = 1536
DISTANCE_METRIC = "COSINE"
location = 'data'
query = 'Which universities did Bertha Kgokong study in?'




def getPDFFiles():
    data_dir = os.path.join(os.curdir,location)
    pdf_files = sorted([x for x in os.listdir(data_dir) if 'DS_Store' not in x])
    return pdf_files,data_dir


##RUN THIS TEST FUNCTION
#redis_client.ft(INDEX_NAME).info()['num_docs']


def createDatabaseIndex():
    # Define RediSearch fields for each of the columns in the dataset
    # This is where you should add any additional metadata you want to capture
    filename = TextField("filename")
    text_chunk = TextField("text_chunk")
    file_chunk_index = NumericField("file_chunk_index")

    # define RediSearch vector fields to use HNSW index

    text_embedding = VectorField(VECTOR_FIELD_NAME,
        "HNSW", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC
        }
    )
    # Add all our field objects to a list to be created as an index
    fields = [filename,text_chunk,file_chunk_index,text_embedding]

    try:
        redis_client.ft(INDEX_NAME).info()
        print(f"Index {INDEX_NAME} already exists")
    except Exception as e:
        redis_client.ft(INDEX_NAME).create_index(fields = fields,
        definition = IndexDefinition(prefix=[PREFIX],
        index_type=IndexType.HASH))
        print(f"Index {INDEX_NAME} was created succesfully")

    return True


##RUN THIS TEST FUNCTION
#redis_client.ft(INDEX_NAME).info()['num_docs']



def addDocumentsToIndex():
    x = getPDFFiles()
    pdf_files = x[0]
    data_dir = x[1]

    # Initialise tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Process each PDF file and prepare for embedding
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir,pdf_file)

        # Extract the raw text from each PDF using textract
        text = textract.process(pdf_path, method='pdfminer')

        # Chunk each document, embed the contents and load to Redis
        handle_file_string((pdf_file,text.decode("utf-8")),tokenizer,redis_client,VECTOR_FIELD_NAME,INDEX_NAME)


##RUN THIS TEST FUNCTION
#redis_client.ft(INDEX_NAME).info()['num_docs']



def queryRedisDatabase():
    result_df = get_redis_results(redis_client,query,index_name=INDEX_NAME)
    redis_result = result_df['result'][0]

    messages = []
    messages.append({"role": "system", "content": "Your name is Karabo. You are a helpful assistant."})

    ENGINEERING_PROMPT = f"""
    Answer this question: {query}
    Attempt to answer the question based on this content: {redis_result}
    """
    question = {}
    question['role'] = 'user'
    question['content'] = ENGINEERING_PROMPT
    messages.append(question)

    response = openai.ChatCompletion.create(model=CHAT_MODEL,messages=messages)

    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'

    return answer




def customChatGPTAnswer(the_query):
    result_df = get_redis_results(redis_client,the_query,index_name=INDEX_NAME)
    redis_result = result_df['result'][0]

    messages = []
    messages.append({"role": "system", "content": "Your name is Karabo. You are a helpful assistant."})

    ENGINEERING_PROMPT = f"""
    Answer this question: {the_query}
    Attempt to answer the question based on this content: {redis_result}
    """
    question = {}
    question['role'] = 'user'
    question['content'] = ENGINEERING_PROMPT
    messages.append(question)

    response = openai.ChatCompletion.create(model=CHAT_MODEL,messages=messages)

    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Oops you beat the AI, try a different question, if the problem persists, come back later.'

    return answer


































##
#
