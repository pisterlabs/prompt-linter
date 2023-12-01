from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

db3 = Chroma(
    persist_directory="./langchainPages/db/chroma_db",
    embedding_function=OpenAIEmbeddings(),
)
docs = db3.similarity_search("What is Langchain?", k=3)
print(docs[0])
print("-----")
print(docs[1])
print("-----")
print(docs[2])
