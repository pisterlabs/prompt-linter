from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def db_create (chunks):
  embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
  return FAISS.from_texts(chunks, embeddings)
