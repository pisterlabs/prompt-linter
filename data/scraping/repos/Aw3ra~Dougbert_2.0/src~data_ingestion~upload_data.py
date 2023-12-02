import pinecone
import os
import openai
import json

pinecone_api = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("PINECONE_INDEX_NAME")
openai.api_key = os.getenv("OPENAI_API_KEY")

# initialise Pinecone
pinecone.init(api_key=pinecone_api, environment=pinecone_env)
# create a new index
index = pinecone.Index(index_name=index_name)

# Function for retrieving data from a json file
def retrieve_data_from_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data

# Function for creating vectors to store in Pinecone
# Takes in a prompt and returns a vector
def create_vector(prompt):
    vector = openai.Embedding.create(
            input=prompt,
            engine= "text-embedding-ada-002",
        )   ["data"][0]["embedding"]
    return vector


# Function to upload vectors to Pinecone
# Takes in a list of content
def upload_vectors(content):
    # Create a list of vectors
    vectors = [create_vector(p['content']) for p in content]
    # Create a list of dictionaries including the content
    data = [{'id': str(i), 'values': v, 'metadata' : p} for i, (v, p) in enumerate(zip(vectors, content))]
    # Upload vectors to Pinecone
    for entries in data:
        print(entries['metadata']['link'], entries['metadata']['title'])
    return index.upsert(vectors=data, namespace=index_name)

# Function that takes in a json object and uploads the data to Pinecone
def upload_data(file_name):
    # Retrieve data from json file
    data = retrieve_data_from_json(file_name)
    # Upload data to Pinecone
    return upload_vectors(data)

def delete_all_data(data):
    data = retrieve_data_from_json(data)
    for i in range(len(data)):
        print(index.delete(ids=[str(i)], namespace=index_name))

if __name__ == "__main__":
    # print(delete_all_data("src/data_ingestion/helius/helius.json"))
    print(upload_data("src/data_ingestion/lancer/lancer-docs.json"))