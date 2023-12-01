from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def faiss_search(pkl, question, model_name, num_results):
    hf_embeddings = HuggingFaceEmbeddings(model_name= f"{model_name}")
    search_index = FAISS.deserialize_from_bytes(embeddings = hf_embeddings, serialized = pkl)
    context = search_index.similarity_search_with_score(question, k = num_results)
    return context