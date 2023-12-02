from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os



class Retriever:
    ''' retrieves the most similar document chunk from the index, based on a prompt '''
    def __init__(self):
        key = os.getenv('OPENAI_API_KEY')
        embedding = OpenAIEmbeddings(openai_api_key=key)
        self.index = FAISS.load_local('rag-data/indexed/pdf_index', embedding)


    def retrieve(self, query, k=1):
        doc_chunks = self.index.similarity_search(query, k=k)
        doc_chunks_list_of_str = [" - "+i.page_content for i in doc_chunks]
        return "\n".join(doc_chunks_list_of_str) 
    


# Retriever().retrieve('mission', k=1)


def appendMessageHistoryContext(messages:list, new_context:str):
    ''' adds RAG context to the message history '''

    prompt = f"Please consider the following context:\n{new_context}"
    new_message = {"role": "user", "content": prompt}

    return messages+[new_message]

