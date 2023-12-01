from langchain.memory import ConversationBufferMemory

def get_memory_short():
    from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="history", chat_memory=message_history, output_key="output", input_key="input", return_messages=True)
    return memory

from langchain.vectorstores.azuresearch import AzureSearch
from langchain.memory import VectorStoreRetrieverMemory

def get_vectorstore_azureSearch():

    vector_store = get_vector_store()
    
    retriever = vector_store.as_retriever(search_kwargs=dict(k=5))
    memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history", input_key="input")
    
    return memory

def get_vector_store():
    import os
    cognitive_search_name = os.environ["AZURE_SEARCH_SERVICE_NAME"]
    vector_store_address: str = f"https://{cognitive_search_name}.search.windows.net/"
    index_name: str = os.environ["AZURE_SEARCH_SERVICE_INDEX_NAME"]
    vector_store_password: str = os.environ["AZURE_SEARCH_SERVICE_ADMIN_KEY"]

    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1, client=any)  
    vector_store: AzureSearch = AzureSearch(azure_cognitive_search_name=vector_store_address,  
                                        azure_cognitive_search_key=vector_store_password,  
                                        index_name=index_name,  
                                        embedding_function=embeddings.embed_query)  
    
    return vector_store