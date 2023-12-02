from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.docstore.document import Document

def create_search_index(docs, model_name):
    documents = [Document(page_content= row["text"], metadata= {"page_number": row["page_number"]}) for row in docs.to_dict('records')]
    embedding_model = SentenceTransformerEmbeddings(model_name= f"sentence-transformers/{model_name}")
    search_index = FAISS.from_documents(documents, embedding_model)
    pkl = search_index.serialize_to_bytes()
    return pkl