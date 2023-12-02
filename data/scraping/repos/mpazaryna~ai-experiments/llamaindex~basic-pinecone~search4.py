import os

import openai
import pinecone
from dotenv import load_dotenv

# Load the stored environment variables
load_dotenv()

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_name = "text-embedding-ada-002"

index_name = PINECONE_INDEX  # Replace with your index name

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
pinecone.whoami()
index = pinecone.Index(index_name)
openai.Engine.list()

# Enter your query here.
query = "What do you know about Boston?"

query_embedding = openai.Embedding.create(input=[query], engine=model_name)
embedding_value = query_embedding["data"][0]["embedding"]

res = index.query(embedding_value, top_k=3, include_metadata=True)
for match in res["matches"]:
    # print(match["metadata"]["wiki_titles"])
    print("----")
