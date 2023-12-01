from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def get_vector_store(text_chunks):
    """
    Creates a vector store from text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        langchain.vectorstores.FAISS: Vector store for the text chunks.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
