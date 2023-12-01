from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from env import OPENAI_API_KEY, QDRANT_URL

loader = TextLoader("./data/insurance.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)

Qdrant.from_documents(
  documents=all_splits,
  embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
  url=QDRANT_URL,
  collection_name='insurance'
)