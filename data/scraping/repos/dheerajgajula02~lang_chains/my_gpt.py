import os
import sys

import constants
import chromadb

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA



os.environ["OPENAI_API_KEY"] =  constants.APIKEY
PERSIST= False

query = sys.argv[1]

# print("Reusing index...\n")
# vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# index = VectorStoreIndexWrapper(vectorstore=vectorstore)


# loader = TextLoader("data.txt")
# print(loader)
# index = VectorstoreIndexCreator().from_loaders([loader])

# print(type(index))
# print(index.query(query, llm=ChatOpenAI()))
# chain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model="gpt-3.5-turbo"),
#     retriever=index.vectorstore.as_retriever(search_kwargs={"k":1}),
# )

if PERSIST and os.path.exists("persist"):
    print("reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("data.txt")
    if PERSIST:
        index=VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever= index.vectorstore.as_retriever(search_kwargs={"k":1}),
)

print(chain.run(query))




