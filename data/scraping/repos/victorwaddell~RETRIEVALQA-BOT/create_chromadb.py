# Chroma Doc Retriever Initialization

from langchain.vectorstores import Chroma


def create_chroma_vectordb(texts, embed):
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents = texts, embedding = embed,
                                        persist_directory = persist_directory)
    
    vectordb.persist()
    vectordb = None
    
    vectordb = Chroma(persist_directory = persist_directory, 
                      embedding_function = embed)
    print("ChromaDB created")
    return vectordb