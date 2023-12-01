import pinecone
import openai
from dotenv import load_dotenv
load_dotenv()
import os
PINECONE_KEY = os.getenv('PINECONE_KEY')
# initialize connection to pinecone
pinecone.init(
    api_key=PINECONE_KEY,  # app.pinecone.io (console)
    environment="eu-west1-gcp"  # next to API key in console
)
# check if index already exists (it shouldn't if this is first time)
index_name = 'personalized-bot'
index = pinecone.Index(index_name)
index.describe_index_stats()

