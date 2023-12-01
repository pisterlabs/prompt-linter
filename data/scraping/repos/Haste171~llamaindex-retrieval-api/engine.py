import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage import StorageContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index import ServiceContext
from decouple import config

class Engine:
    def __init__(self) -> None:
        self.openai_api_key = config('OPENAI_API_KEY')
        self.pinecone_api_key = config('PINECONE_API_KEY')
        self.pinecone_environment = config('PINECONE_ENVIRONMENT')
        self.pinecone_index = config('PINECONE_INDEX')
        self.embedding_model = OpenAIEmbedding(embed_batch_size=10, api_key=self.openai_api_key)


    def load(self, loader: object, namespace: str):
        """
        Designed to load or index data/documents from a given loader into a Pinecone vector store, using specific configurations and services, such as OpenAI's model for processing and embedding the data.
        """
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
        pinecone_index = pinecone.Index(self.pinecone_index)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="content", namespace=namespace)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613", api_key=self.openai_api_key), 
                                                       embed_model=self.embedding_model, chunk_size=1000)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(loader, service_context=service_context, storage_context=storage_context)


    def query(self, prompt: str, namespace: str, temperature: float):
        """
        Returns a response to the provided prompt based on the indexed data and the OpenAI model.
        Queries Pinecone vector store using OpenAI, and retrieve relevant responses based on the input prompt.
        """
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_environment)
        pinecone_index = pinecone.Index(self.pinecone_index)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613", temperature=temperature, api_key=self.openai_api_key), 
                                                       embed_model=self.embedding_model, chunk_size=1000)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index, text_key="content", namespace=namespace)
        # retriever = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context).as_retriever(similarity_top_k=1)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
        query_engine = index.as_query_engine(chat_mode="openai")
        response = query_engine.query(prompt)
        return response
    

    # def chat(self, prompt: str):
    #     service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613", api_key=self.openai_api_key), 
    #                                                    embed_model=self.embedding_model, chunk_size=1000)
    #     data = SimpleDirectoryReader(input_dir="data").load_data()
    #     index = VectorStoreIndex.from_documents(data, service_context=service_context)
    #     chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True, similarity_top_k=5, streaming=True)
    #     response = chat_engine.chat(prompt, function_call="query_engine_tool")
    #     return response
    
