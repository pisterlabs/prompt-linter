from langchain.chains.router import MultiRetrievalQAChain
from langchain.llms import OpenAI

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

sou_docs = TextLoader('../../state_of_the_union.txt').load_and_split()
sou_retriever = FAISS.from_documents(sou_docs, OpenAIEmbeddings()).as_retriever()

pg_docs = TextLoader('../../paul_graham_essay.txt').load_and_split()
pg_retriever = FAISS.from_documents(pg_docs, OpenAIEmbeddings()).as_retriever()

personal_texts = [
    "I love apple pie",
    "My favorite color is fuchsia",
    "My dream is to become a professional dancer",
    "I broke my arm when I was 12",
    "My parents are from Peru",
]
personal_retriever = FAISS.from_texts(personal_texts, OpenAIEmbeddings()).as_retriever()

retriever_infos = [
    {
        "name": "state of the union", 
        "description": "Good for answering questions about the 2023 State of the Union address", 
        "retriever": sou_retriever
    },
    {
        "name": "pg essay", 
        "description": "Good for answering questions about Paul Graham's essay on his career",
        "retriever": pg_retriever
    },
    {
        "name": "personal", 
        "description": "Good for answering questions about me", 
        "retriever": personal_retriever
    }
]

chain = MultiRetrievalQAChain.from_retrievers(OpenAI(), retriever_infos, verbose=True)

print(chain.run("What did the president say about the economy?"))

print(chain.run("What is something Paul Graham regrets about his work?"))

print(chain.run("What is my background?"))