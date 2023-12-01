import dotenv

dotenv.load_dotenv()

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

doc_list = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
    "I like computers by Apple",
    "I love fruit juice"
]

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 2

result = bm25_retriever.get_relevant_documents("Apple")
print("apple", result)

result = bm25_retriever.get_relevant_documents("a green fruit")
print("a green fruit", result)

result = bm25_retriever.dict
print("dict", result)

# Embeddings - Dense retrievers FAISS

faiss_vectorstore = FAISS.from_texts(doc_list, embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})
result = faiss_retriever.get_relevant_documents("A green fruit")
print("A green fruit from dense retriever", result)

# Ensemble Retriever

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,
                                                   faiss_retriever],
                                       weights=[0.5, 0.5])

docs = ensemble_retriever.get_relevant_documents("A green fruit")
print("from the ensemble retriever", docs)

docs = ensemble_retriever.get_relevant_documents("Apple Phones")
print(docs)
