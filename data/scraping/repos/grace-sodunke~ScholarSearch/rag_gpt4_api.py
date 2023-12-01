import time
import os
import pandas as pd
import numpy as np
import openai
import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from huggingface_hub import login
from milvus import default_server
from pymilvus import (
    connections, utility
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

default_server.stop()
default_server.cleanup()

start_time = time.time()
default_server.start() 
end_time = time.time()
print(f"Milvus server startup time: {end_time - start_time} sec")

time.sleep(15) 
connections.connect(host='127.0.0.1', 
                  port=default_server.listen_port,
                  show_startup_banner=True)
print(utility.get_server_version()) 

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 413
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"device: {DEVICE}")

openai.api_key = os.getenv("OPENAI_API_KEY")
hub_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token=hub_token)
model_name = "BAAI/bge-base-en-v1.5"
retriever = SentenceTransformer(model_name, device=DEVICE)
print(type(retriever))
print(retriever)

MAX_SEQ_LENGTH = retriever.get_max_seq_length() 
HF_EOS_TOKEN_LENGTH = 1
EMBEDDING_LENGTH = retriever.get_sentence_embedding_dimension()
print(f"model_name: {model_name}")
print(f"EMBEDDING_LENGTH: {EMBEDDING_LENGTH}")
print(f"MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

model_kwargs = {"device": DEVICE}
encode_kwargs = {'normalize_embeddings': True}
lc_encoder = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
type(lc_encoder)

file_path1 = "/Users/gracesodunke/Documents/Google STEP/LONDON Grace Ayomide Sodunke_Offer Letter_463737234_1688056780235.pdf"
file_path2 = "/Users/gracesodunke/Documents/Google STEP/Grace Ayomide Sodunke_Offer Letter_463737234_1683043578334_20230509-1541.pdf" 

MILVUS_PORT = 19530
MILVUS_HOST = "127.0.0.1"

vector_store = Milvus(
    collection_name="Collection3",
    embedding_function=lc_encoder,
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
)

def upload_document(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    chunk_size = MAX_SEQ_LENGTH - HF_EOS_TOKEN_LENGTH
    chunk_overlap = np.round(chunk_size * 0.10, 0)
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
    )
    chunks = text_splitter.create_documents(
        [p.page_content for  p in pages], 
        metadatas=[{"name": pdf_path} for p in pages])
    end_time = time.time()
    print(f"chunking time: {end_time - start_time}")
    print(f"type: {type(chunks)}, len: {len(chunks)}, type: {type(chunks[0])}")
    print(f"type: list of {type(chunks[0])}, len: {len(chunks)}") 

    print()
    print("Looking at a sample chunk...")
    print(chunks[0].metadata)
    print(chunks[0].page_content)

    print("Start inserting entities")
    start_time = time.time()
    vector_store.add_documents(chunks)
    end_time = time.time()
    print(f"LlamaIndex Milvus insert time for {len(chunks)} vectors: {end_time - start_time} seconds")
    print(f"type: {type(vector_store)}")

upload_document(file_path1)
upload_document(file_path2)

question = 'What is the salary per annum?'
query = [question]
QUERY_LENGTH = len(query[0])
print(f"query length: {QUERY_LENGTH}")
# METADATA_URL = ""
# SEARCH_PARAMS = dict({
#     "expr": "text = METADATA_URL",
#     })
start_time = time.time()
docs = vector_store.similarity_search(
    question,
    k=4,
    param=None,
    verbose=True,
    )
end_time = time.time()
print(f"Milvus query time: {end_time - start_time}")
for d in docs:
    print(d.metadata)
    print(d.page_content[:100])
print(f"Count raw retrievals: {len(docs)}")

unique_sources = []
unique_texts = []
for doc in docs:
    if doc.page_content not in unique_texts:
        unique_texts.append(doc.page_content)
        unique_sources.append(doc.metadata)
print(f"Count unique texts: {len(unique_texts)}")
#[ print(text) for text in unique_texts ]

formatted_context = list(zip(unique_sources, unique_texts))
context = ""
for source, text in formatted_context:
    context += f"{text} "
print(len(context))

p = f"""You are a helpful and knowledgeable agent. To achieve your goal of answering a query
    correctly, you have access to contextual information. You are given a query and must produce a concise summary
    of maximum 500 words that uses the relevant information from the context, given in a grammatically correct and
    readable format.

    Query: {query}
    Context: {context}
    Your answer: """
result = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    prompt=p,
    max_tokens=1000,
    temperature=0
    )
answer = result['choices'][0]['text'].lstrip('\n')

print(answer)

def query_search(question):
    query = [question]
    QUERY_LENGTH = len(query[0])
    print(f"query length: {QUERY_LENGTH}")
    start_time = time.time()
    docs = vector_store.similarity_search(
        question,
        k=4,
        param=None,
        verbose=True,
        )
    end_time = time.time()
    print(f"Milvus query time: {end_time - start_time}")
    for d in docs:
        print(d.metadata)
        print(d.page_content[:100])
    print(f"Count raw retrievals: {len(docs)}")

    unique_sources = []
    unique_texts = []
    for doc in docs:
        if doc.page_content not in unique_texts:
            unique_texts.append(doc.page_content)
            unique_sources.append(doc.metadata)
    print(f"Count unique texts: {len(unique_texts)}")
    formatted_context = list(zip(unique_sources, unique_texts))
    context = ""
    for source, text in formatted_context:
        context += f"{text} "
    print(len(context))

    p = f"""You are a helpful and knowledgeable agent. To achieve your goal of answering a query
    correctly, you have access to contextual information. You are given a query and must produce a concise summary
    of maximum 500 words that uses the relevant information from the context, given in a grammatically correct and
    readable format.

    Query: {query}
    Context: {context}
    Your answer: """
    result = openai.Completion.create(
        model="gpt-3.5-turbo-16k",
        prompt=p,
        max_tokens=1000,
        temperature=0
        )
    answer = result['choices'][0]['text'].lstrip('\n')

    print(answer)
    return {"summary": answer}

#def practice_topic