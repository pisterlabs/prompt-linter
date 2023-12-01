import os

# from langchain.chat_models import ChatOpenAI
from langchain.llms import Cohere
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from env import MONGO_URI, COHERE_API_KEY, DB_NAME, COLLECTION_NAME

class Question(BaseModel):
    __root__: str

# Mongo Atlas Vector Search
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default" # indexer configured on Atlas
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]

# Embedder
embedder = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

# Read from MongoDB Atlas Vector Search
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embedder,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = Cohere(cohere_api_key=COHERE_API_KEY)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)
chain = chain.with_types(input_type=Question)

def _ingest(url: str) -> dict:
    # Load docs
    loader = PyPDFLoader(url)
    data = loader.load()

    # Split docs
    print("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)

    # Insert the documents in MongoDB Atlas Vector Search
    print("Inserting documents in MongoDB Atlas Vector Search")
    _ = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embedder,
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return {}

ingest = RunnableLambda(_ingest)


# query = "What were the compute requirements for training GPT 4"
# results = vector_search.similarity_search(query)

# print(results[0].page_content)

# query = "gpt-4"
# results = vector_search.similarity_search(
#     query=query,
#     k=20,
# )

# # Display results
# #print(dict(results[0].metadata).keys())
# for result in results:
#     print( result)


