from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from settings import llm
import os

embeddings = OpenAIEmbeddings()

vectorstore = Pinecone.from_existing_index(os.environ.get("PINECONE_INDEX_NAME"), embeddings,namespace='namespace1')

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=vectorstore.as_retriever()
)

res = qa("本文的主要内容是什么？")
print(res)
