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

#textsplitter-----------------keep large chunk size and miniumum overlap to get new sentences in each chunk

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

#loading vectors into vector db-----------------

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
    
    
query = "what is mentioned about barve dental?"
docs = db.similarity_search(query, k=4)

#print(docs[0].page_content)
#print(docs[1].page_content)
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

llm = OpenAI(model="text-davinci-003", temperature=0.7, max_tokens=500)

#llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"max_length":7692, "max_new_tokens":500, "temperature":0.7})


chain = LLMChain(llm=llm, prompt=prompt, output_key= "testi")
response = chain.run({"context": docs})
print(response)

#custom tools-----------------

from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class CustomSearchTool(BaseTool):
    name = "content_writing_tool"
    description = "useful for summarising the content about my dental clinic"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        store = vectorstore.as_retriever()
        docs = store.get_relevant_documents(query)
        text_list = [doc.page_content for doc in docs]
        return "\n".join(text_list)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
#-----------------

from langchain.agents import AgentType

from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.agents import initialize_agent


#either use this single custom tool as below----
tools = [CustomSearchTool(), llm]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("write detailed understanding about barve dental?")

#-----

#or use the list of tools as below----

from langchain.agents import Tool

from langchain.agents import AgentType


tool_list = [
     Tool(
          name = "llm-math",
          func=tools[0].run,
          description="Useful for when you need to answer questions about math"
      ),
      Tool(
           name= "serpapi",
         func=tools[0].run,
          description= "A search engine. Useful for when you need to answer questions about current events"
        ),
     Tool(
         name= "content_writer",
         func=tools[0].run,
         description= "useful for summarising the content about my dental clinic"
     ),    
]

agent = initialize_agent(tool_list, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("what is the website of barve dental?")

#------









