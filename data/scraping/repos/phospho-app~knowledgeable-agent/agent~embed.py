import os
import pinecone
import openai
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

print("### Knowledgeable agent from phospho ###")
print("")

try:

    # Load env variables from .env file
    load_dotenv()

    # Get env variables
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENV = os.getenv('PINECONE_ENV')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    openai.api_key = OPENAI_API_KEY

except Exception as e:
    print("Error loading env variables. Did you create a .env file?")
    print(e)
    exit(1)

loader = TextLoader('tmp/text.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

print("> generated embeddings")

docsearch = Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX_NAME)

print("> created index")

print("> next step: deploy using phospho")