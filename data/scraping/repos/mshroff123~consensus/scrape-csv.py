import csv
import os
from tqdm.auto import tqdm
import pinecone
import json, codecs
import numpy as np
from langchain.document_loaders import DataFrameLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import datetime
import openai


openai.api_key = "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"

file_path = './submissions_filtered.csv'
questions = []
with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Skip the header row if present
    next(csv_reader, None)
    
    # Read and print the first 5 lines
    for row in csv_reader:
        # add the question to the questions dataset
        questions.append(row[1])


# remove duplicates
questions = list(set(questions))
print('\n'.join(questions[:5]))
print(len(questions))




# model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


# init pinecone
PINECONE_API_KEY = '974f9758-d34f-4083-b82d-a05e3b1742ae'
PINECONE_ENV = 'us-central1-gcp'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)


# create an index

index_name = 'semantic-search-6998'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine'
    )

# now connect to the index
index = pinecone.Index(index_name)


batch_size = 128


for i in tqdm(range(0, len(questions), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(questions))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadatas = [{'text': text} for text in questions[i:i_end]]
    # create embeddings
    # xc = model.encode(questions[i:i_end])
    openai_embed = OpenAIEmbeddings(openai_api_key=openai.api_key)

    xc = openai_embed.embed_documents(questions[i:i_end])
    # create records list for upsert
    records = []
    for i in range(len(ids)):
        record = (ids[i], xc[i], metadatas[i])
        records.append(record)



    # upsert to Pinecone
    print("uploading")
    index.upsert(vectors=records)

# check number of records in the index

print(index.describe_index_stats())