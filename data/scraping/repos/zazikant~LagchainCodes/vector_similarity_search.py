import os
import openai

import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
API_KEY = os.environ.get("API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
openai.api_key = os.environ["OPENAI_API_KEY"]

#Loaders----------------

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader(
    "./docu", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
)

documents = loader.load()

#textsplitter-----------------keep large chunk size and miniumum overlap to get new 

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=2,
)

docs = text_splitter.split_documents(documents)

#embeddings-----------------

from langchain.embeddings import OpenAIEmbeddings
openai_embeddings = OpenAIEmbeddings()


# from langchain.embeddings import HuggingFaceEmbeddings
# openai_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


from langchain.vectorstores.faiss import FAISS

import pickle

#Very important - db below is used by similarity search and not been used for agents

db = FAISS.from_documents(docs, openai_embeddings)

#Very important - vectorstore below is used by agents in tools and not been used for similarity search

vectorstore = FAISS.from_documents(documents, openai_embeddings)

#dump and load the vector store for vector store that is going to be used by agents-----

import pickle

with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
    
with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)
    
#dump and load the vector store for vector store that is going to be used by similarity search-----

import pickle

with open("db.pkl", "wb") as f:
    pickle.dump(db, f)
    
with open("db.pkl", "rb") as f:
    db = pickle.load(f)

query = "what is mentioned about Laser-assisted gum treatments?"
docs = db.similarity_search(query, k=4)

# print(docs[0].page_content)
# print(docs[1].page_content)
# print(docs[2].page_content)
# print(docs[3].page_content)



import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import ConversationSummaryBufferMemory


template = """
Summarize the key points from these documents.

text: {context}
"""

prompt  = PromptTemplate(
    input_variables=["context"],
    template=template
)

llm = OpenAI(model="text-ada-001", temperature=0.7, max_tokens=500)

chain = LLMChain(llm=llm, prompt=prompt, output_key= "testi")
response = chain.run({"context": docs})
print(response)


chain.predict(input="I ordered Pizza Salami and it was awesome!")





