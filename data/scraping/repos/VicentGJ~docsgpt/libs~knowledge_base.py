from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store
