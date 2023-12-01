import pinecone
from decouple import config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


embeddings_api = OpenAIEmbeddings(openai_api_key=config("OPENAI_API_KEY"))

pinecone.init(api_key=config("PINECONE_API_KEY"), environment="gcp-starter")
pinecone_index = pinecone.Index("langchain-vector-store")
vectorstore = Pinecone(pinecone_index, embeddings_api, "text")


query = "What is the fastest way to get rich?"
print(vectorstore.similarity_search(query, k=5))
