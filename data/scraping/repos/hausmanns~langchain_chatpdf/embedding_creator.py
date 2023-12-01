from langchain.embeddings.openai import OpenAIEmbeddings  # Wrapper for embeddings from OpenAI
from langchain.vectorstores import FAISS  # Wrapper for FAISS

def create_embeddings(chunks):
    """
    Function to create embeddings using Langchain's OpenAIEmbeddings and FAISS.

    Parameters:
    chunks (list): List of chunks to create embeddings.

    Returns:
    knowledge_base (object): FAISS object with the embeddings.
    """
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base
