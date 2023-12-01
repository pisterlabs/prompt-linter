from dotenv import load_dotenv
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
NAME_SPACE = os.getenv("NAME_SPACE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)
embeddings = OpenAIEmbeddings(
    disallowed_special=(),
)
docsearch = Pinecone.from_existing_index(
    PINECONE_INDEX, embeddings, namespace=NAME_SPACE
)
docsearch.delete(namespace=NAME_SPACE, delete_all=True)
