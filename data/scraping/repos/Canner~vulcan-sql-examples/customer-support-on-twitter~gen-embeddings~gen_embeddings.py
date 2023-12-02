import json
import os

from dotenv import load_dotenv
from langchain.embeddings import CohereEmbeddings
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


load_dotenv()


print('Loading data from CSV file')
df = pd.read_csv('../twcs.csv')
# since we are using the free Cohere API, we need to limit the number of requests
df = df[:5000]


print('Adding the text_embeddings column to the dataframe')
embeddings_model = CohereEmbeddings(cohere_api_key=os.environ.get('COHERE_API_KEY'))
# dimensions of the embeddings are 4096
docs_embeddings = embeddings_model.embed_documents(df['text'].to_list())

for i, doc_embeddings in enumerate(docs_embeddings):
    docs_embeddings[i] = json.dumps(doc_embeddings)

df['text_embeddings'] = docs_embeddings


print('Inserting data to the PostgreSQL database')
conn = psycopg2.connect(
    dbname=os.environ.get('PG_DATABASE'),
    user=os.environ.get('PG_USER'),
    password=os.environ.get('PG_PASSWORD'),
    host=os.environ.get('PG_HOST'),
    port=os.environ.get('PG_PORT')
)

cursor = conn.cursor()

data = [
    (
        row.tweet_id,
        row.author_id,
        row.inbound,
        row.created_at,
        row.text,
        row.text_embeddings,
        row.response_tweet_id,
        row.in_response_to_tweet_id
    )
    for row in df.itertuples()
]

execute_values(
    cursor,
    """
    INSERT INTO twcs 
    (
        tweet_id, 
        author_id, 
        inbound, 
        created_at, 
        text, 
        text_embeddings, 
        response_tweet_id, 
        in_response_to_tweet_id
    )
    VALUES %s
    """,
    data
)

conn.commit()
cursor.close()
conn.close()
print('DONE')
