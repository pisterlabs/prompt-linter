# Loading documents from a directory with LangChain
from langchain.document_loaders import DirectoryLoader

directory = 'data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)

# Splitting documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)

# Creating embeddings
from langchain.embeddings import SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

query_result = embeddings.embed_query("Hello world")

#Storing embeddings in Pinecone 
import pinecone 
from langchain.vectorstores import Pinecone
pinecone.init(
    api_key="",  # find at app.pinecone.io
    environment=""  # next to api key in console
)
index_name = "langchain-chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)




