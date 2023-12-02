import openai
import chromadb
import os

openai.api_key = os.environ['OPEN_API_KEY']
chroma_client = chromadb.PersistentClient(path="chromadb_data")


def get_chroma_collection():
    collection_name = "minojiro_info"
    try:
        return chroma_client.get_collection(collection_name)
    except ValueError:
        return chroma_client.create_collection(collection_name)


def get_embeddings(text):
    res = openai.Embedding.create(
        model='text-embedding-ada-002',
        input=text,
    )
    return res['data'][0]['embedding']
