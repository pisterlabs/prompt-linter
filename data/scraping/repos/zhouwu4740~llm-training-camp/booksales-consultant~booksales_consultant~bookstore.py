from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings


def bookstore_retriever():
    db = FAISS.load_local('../jupyter/books', OpenAIEmbeddings())
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.7, "k": 3})
    return retriever


BOOK_STORRE_RETRIEVER = bookstore_retriever()
