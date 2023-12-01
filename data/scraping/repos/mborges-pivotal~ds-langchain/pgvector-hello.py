# pgvector
#
#
# note: Interesting functions to calculate the cost of LLM model
#
import openai
import os
import pandas as pd
import numpy as np
import json
import tiktoken
import psycopg2
import ast
import pgvector
import math
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# Get openAI api key by reading local .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY'] 

# Load your CSV file into a pandas DataFrame
df = pd.read_csv('./data/blog_posts_data.csv')
df.head()

conn = psycopg2.connect(
    database="marceloborges",
    user="marceloborges",
    password="",
    host="localhost",
    port= '5432'
)

cur = conn.cursor()

#install pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector");
conn.commit()

# Register the vector type with psycopg2
register_vector(conn)

# Create table to store embeddings and metadata
table_create_command = """
CREATE TABLE embeddings (
            id bigserial primary key, 
            title text,
            url text,
            content text,
            tokens integer,
            embedding vector(1536)
            );
            """

cur.execute(table_create_command)
cur.close()
conn.commit()

#Batch insert embeddings and metadata from dataframe into PostgreSQL database
register_vector(conn)
cur = conn.cursor()
# Prepare the list of tuples to insert
data_list = [(row['title'], row['url'], row['content'], int(row['tokens']), np.array(row['embeddings'])) for index, row in df_new.iterrows()]
# Use execute_values to perform batch insertion
execute_values(cur, "INSERT INTO embeddings (title, url, content, tokens, embedding) VALUES %s", data_list)
# Commit after we insert all embeddings
conn.commit()

cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
num_records = cur.fetchone()[0]
print("Number of vector records in table: ", num_records,"\n")
# Correct output should be 129

# print the first record in the table, for sanity-checking
cur.execute("SELECT * FROM embeddings LIMIT 1;")
records = cur.fetchall()
print("First record in table: ", records)

#######################################
# Helper functions to help us create the embeddings
#######################################

# Helper func: calculate number of tokens
def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Helper function: calculate length of essay
def get_essay_length(essay):
    word_list = essay.split()
    num_words = len(word_list)
    return num_words

# Helper function: calculate cost of embedding num_tokens
# Assumes we're using the text-embedding-ada-002 model
# See https://openai.com/pricing
def get_embedding_cost(num_tokens):
    return num_tokens/1000*0.0001

# Helper function: calculate total cost of embedding all content in the dataframe
def get_total_embeddings_cost():
    total_tokens = 0
    for i in range(len(df.index)):
        text = df['content'][i]
        token_len = num_tokens_from_string(text)
        total_tokens = total_tokens + token_len
    total_cost = get_embedding_cost(total_tokens)
    return total_cost

# quick check on total token amount for price estimation
total_cost = get_total_embeddings_cost()
print("estimated price to embed this content = $" + str(total_cost))
      
    
