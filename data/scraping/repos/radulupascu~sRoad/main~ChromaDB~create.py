from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import os

from langchain.text_splitter import CharacterTextSplitter

try:
  API_KEY = open("../API_KEY", "r").read()
except FileNotFoundError:
  pass

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
if (os.path.exists('../db') == False):
  persist_directory = '../db'

  ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
  embedding = OpenAIEmbeddings(openai_api_key=API_KEY)
                                  
  loader = CSVLoader(file_path='../Combined_Courses.csv')
  documents = loader.load()

  text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
  texts = text_splitter.split_documents(documents=documents)

  vectordb = Chroma.from_documents(documents=documents, 
                                  embedding=embedding,
                                  persist_directory=persist_directory)

                                

  # persiste the db to disk
  vectordb.persist()
  vectordb = None
else:
  print("Database already exists")


