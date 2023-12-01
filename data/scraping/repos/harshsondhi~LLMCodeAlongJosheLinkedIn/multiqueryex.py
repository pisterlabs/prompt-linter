import sys

import chromadb
import pandas
import sqlite3
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WikipediaLoader
from langchain.retrievers.multi_query import MultiQueryRetriever
import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
chat = ChatOpenAI(openai_api_key=api_key, temperature=0)


loader = WikipediaLoader(query="MKUltra")
documents = loader.load()
print(len(documents))

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function, persist_directory="./mk_ultra")
db.persist()

from langchain.retrievers.multi_query import MultiQueryRetriever

question = "When was this declassified?"
llm1 = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm1)
# print(os.path.dirname(sys.executable))

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)
print(unique_docs[0].page_content)
