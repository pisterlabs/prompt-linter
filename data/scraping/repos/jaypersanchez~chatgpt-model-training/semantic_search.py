import pandas as pd
import numpy as np
import openai 
import os
# Import psycopg2 for database connectivity - Postgresql
import psycopg2
import json
import csv

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
# Token Tuning 
token_tuning_params = {
    "num_tokens": 4, # the number of tokens to use
    "max_tokens": 10, # the maximum number of tokens to generate
    "min_token_length": 4, # the minimum length of each token
    "max_token_length": 8 # the maximum length of each token
}

# Initialize the OpenAI class
openai.organization = os.environ['OPENAI_ORG_KEY']
openai.api_key = os.environ['OPENAI_API_KEY']

#check if we are authenticated
modelList = openai.Engine.list()
#print out the available data model for embedding
#print(modelList)

# load the csv file 
csv_data = [] 
counter = 1
#need to cleanup data
with open('./models/exchange.csv', 'r', encoding='utf-8') as csvfile: 
    reader = csv.reader(csvfile) 
    table = []
    counter = 0
    for row in reader:
        #read only up to first 100 rows
        if counter <= 100: 
            csv_data = row
            res = openai.Embedding.create(data=csv_data, model=embedding_model,input=csv_data, params=max_tokens)
            embedding_list = [item["embedding"] for item in res["data"]]
            table.append([csv_data, embedding_list[0]])
            counter += 1
#print(table)

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect("dbname=Satoshi user=postgres password=Ph1LL!fe host=localhost")

# Create a cursor object
cursor = conn.cursor()
for row in table:
    csv_data = row[0]
    embedding_list = row[1]
    cursor.execute("INSERT INTO vectors (csv_data, embedding_list) VALUES (%s, %s)", (csv_data, embedding_list))

conn.commit()
conn.close()
print("Embedding Done")
