from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Epsilla
from pyepsilla import vectordb
from sentence_transformers import SentenceTransformer

from typing import List
from glob import glob



# Local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get list of all files in "./documents/"
files = glob("./documents/*")


class LocalEmbeddings():
  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return model.encode(texts).tolist()
  

embeddings = LocalEmbeddings()


for file in files:
  loader = TextLoader(file)
  documents = loader.load()
  splitted_documents = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200).split_documents(documents)
  print("working")
  client = vectordb.Client()
  vector_store = Epsilla.from_documents(
    splitted_documents,
    embeddings,
    client,
    db_path="/tmp/localchatdb",
    db_name="LocalChatDB",
    collection_name="LocalChatCollection"
  )

