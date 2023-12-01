from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.faiss.FAISS.html

documents = [
    "Adam is a programmer who specializes in JavaScript full-stack development",
    "with a particular focus on using frameworks like Svelte and NestJS.",
    "Adam has a dog named Alexa.",
    "Adam is also a designer.",
]

documents = [Document(page_content=doc) for doc in documents]
embeddings = OpenAIEmbeddings()

faiss = FAISS.from_documents(documents, embeddings)

retrieved_docs = faiss.similarity_search(query="What does Adam do", k=3)

for rd in retrieved_docs:
    print(rd)
