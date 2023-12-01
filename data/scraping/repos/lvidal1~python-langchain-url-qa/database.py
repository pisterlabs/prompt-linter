import openai

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


VECTOR_FILE = "chroma_db"

def get_vector_db(key, documents, persist=False):
  openai.api_key = key
  embeddings = OpenAIEmbeddings()

  if persist:
    db = Chroma.from_documents(documents, embeddings, persist_directory=VECTOR_FILE)
    db.persist()
  else:
    db = Chroma(persist_directory=VECTOR_FILE, embedding_function=embeddings)

  return db

def search(db, query, results=10):
  return db.similarity_search(
    query=query,
    n_results=results
)