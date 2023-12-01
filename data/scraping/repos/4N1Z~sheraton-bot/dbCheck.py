import qdrant_client
import streamlit as st

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Qdrant

QDRANT_HOST = st.secrets["QDRANT_HOST"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# QDRANT_HOST = "https://046843ec-e71e-4ab6-9100-711d8c1ee55b.us-east4-0.gcp.cloud.qdrant.io"
# QDRANT_API_KEY = "u-g6fMdlAHQzNEH16uAZrkB1Fw1N4W9pqLwqrXJYwAWRsk2aHCzn3w" 


# Creating a persistant DB
client = qdrant_client.QdrantClient(
    url = QDRANT_HOST,
    api_key= QDRANT_API_KEY,
)


# create_collection
collection_name = "hotelDataCollection"
vector_config = qdrant_client.http.models.VectorParams(
    size = 4096,
    distance = qdrant_client.http.models.Distance.COSINE
    # distance = qdrant_client.http.models.Distance.DOT
)
client.recreate_collection(
    collection_name = collection_name,
    vectors_config = vector_config,
)
# print(client)

#  add vectors
web_links = ["https://hotels-ten.vercel.app/api/hotels"] 
loader = WebBaseLoader(web_links)
document=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(document)

embeddings = CohereEmbeddings(model = "embed-english-v2.0")
print(" embedding docs !")

vector_store = Qdrant(
    client=client,
    collection_name = collection_name,
    embeddings=embeddings
)
# vector_store.add_documents(texts)
vector_store.add_documents(texts)
retriever=vector_store.as_retriever()
# print(retriever)


