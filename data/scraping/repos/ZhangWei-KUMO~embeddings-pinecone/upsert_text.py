from settings import get_embedding,index
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
import pinecone
import os

pinecone.init(
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
    api_key=os.environ.get("PINECONE_API_KEY")
)
embeddings = OpenAIEmbeddings()
loader = TextLoader("./example.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
index_name = "gpt-test-pdf"

meta = {
    "genre":"news",
    "years":2023
}

Pinecone.from_documents(documents=docs, embedding=embeddings, index_name=index_name,namespace='qingang',)

