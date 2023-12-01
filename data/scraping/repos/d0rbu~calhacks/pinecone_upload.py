import time
from datetime import datetime
import pandas as pd
import pinecone
import openai
import os
from dotenv import load_dotenv
load_dotenv()

csv = pd.read_csv(
    '/Users/sunnyjay/Documents/vscode/Hackathon/calhacks/calhacks_backend/emails.csv')
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# def get_embedding(text):
#     response = openai.Embedding.create(
#         input=text,
#         model="text-embedding-ada-002"
#     )
#     return response['data'][0]['embedding']


pinecone.init(api_key=PINECONE_API_KEY,
              environment="eu-west1-gcp")
index = pinecone.Index("calhacks")


def embed_and_upsert(df):

    if len(df) == 0:
        return
    to_be_embedded = []
    for i in range(len(df)):
        text = f"""
        Sender: {df['sender'][i]}
        Subject: {df['subject'][i]}
        Body: {df['body'][i]}
        """
        to_be_embedded.append(text)

    response = openai.Embedding.create(
        input=to_be_embedded,
        model="text-embedding-ada-002"
    )
    embeds = [record['embedding'] for record in response['data']]

    print('embeddings done')

    vectors = []
    for i in range(len(df)):
        vector_id = str(i+len(csv))
        embed = embeds[i]
        metadata = {
            'sender': df['sender'][i],
            'subject': df['subject'][i],
            'body': df['body'][i],
            'date': df['date'][i]
        }
        vectors.append((vector_id, embed, metadata))

    print(vectors)
    print('done assembing upserts')

    # for i in range(0,100,20):
    #     to_upload = vectors[i:i+20]
    #     res = index.upsert(
    #         vectors=to_upload
    #     )
    #     print(res)
    #     time.sleep(1)

    res = index.upsert(
        vectors=vectors
    )
    print(res)


def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

    
def query(query):
    embedding = get_embedding(query)
    query_response = index.query(
        top_k=10,
        include_values=True,
        include_metadata=True,
        vector=embedding
    )
    return query_response