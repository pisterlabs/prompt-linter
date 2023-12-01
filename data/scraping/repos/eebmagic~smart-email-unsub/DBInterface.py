import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os

load_dotenv()

client = chromadb.PersistentClient('db')


openai_token = os.getenv('OPENAI_TOKEN')
openai_embed = OpenAIEmbeddingFunction(
    api_key=openai_token
)

collection = client.get_or_create_collection(
    name="TrashMail",
    embedding_function=openai_embed
)
