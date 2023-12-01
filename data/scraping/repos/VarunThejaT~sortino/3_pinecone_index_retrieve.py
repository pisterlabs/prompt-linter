import os
import pinecone
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-embedding-ada-002"

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="northamerica-northeast1-gcp"  # find next to API key in console
)

index = pinecone.Index('openai')
query = "What is the revenue of 'Family of Apps' division of Meta in 2022?"

# create the query embedding
xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']

# query, returning the top 5 most similar results
res = index.query([xq], top_k=3, include_metadata=True, namespace="meta")

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
