import pinecone
import openai
import os

# Pinecone settings
index_name =  os.getenv("PINECONE_INDEX_NAME")

# OpenAI settings
embed_model = "text-embedding-ada-002"

# Connect to Pinecone
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT")
pinecone.init(api_key=api_key, environment=env)

# Initialize OpenAI embedding engine
embedding = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

# Create Pinecone database
if index_name not in pinecone.list_indexes():
    print("Creating pinecone index: " + index_name)
    pinecone.create_index(
        index_name,
        dimension=len(embedding['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['source', 'id']}
    )
