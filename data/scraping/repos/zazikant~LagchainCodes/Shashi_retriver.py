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

#Loaders----------------

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader(
    "./FAQ", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
)

docs = loader.load_and_split()

#-----------------

#textsplitter-----------------

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10,
)

documents = text_splitter.split_documents(docs)
documents[0]

#-----------------

#embeddings-----------------

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

#-----------------

#loading vectors into vector db-----------------

from langchain.vectorstores.faiss import FAISS
import pickle

vectorstore = FAISS.from_documents(documents, embeddings)

with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
    
#-----------------

#loading the database-----------------

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

#-----------------

#prompt-----------------

from langchain.prompts import PromptTemplate

prompt_template = """You are a marketing manager, and you are going to pitch to customer. You have a FAQ document that you want to use to answer questions. You have a question from the customer and you want to answer it using the FAQ document.

{context}

Question: {question}
Answer here:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#-----------------

#chains-----------------

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

chain_type_kwargs = {"prompt": PROMPT}

llm = OpenAI(openai_api_key=API_KEY)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

quesst = "who is shashikant?"
qa.run({"query": quesst})
qa.run({"query": "what is his height?"})
qa.run({"query": "sing a poem about him"})

#-----------------

#memory-----------------

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

#-----------------

#use memory in chain-----------------

from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY),
    memory=memory,
    retriever=vectorstore.as_retriever(),
    combine_docs_chain_kwargs={"prompt": PROMPT},
)


query = "what does shashikant play?"
qa({"question": query})
qa({"question": "what is his likings?"})

#-----------------

#loading vectordb--------

import pickle

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)
    
#-----------------

#custom tools-----------------

from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class CustomSearchTool(BaseTool):
    name = "presenter"
    description = "useful tool for writing powerful presentations"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        store = vectorstore.as_retriever()
        docs = store.get_relevant_documents(query)
        text_list = [doc.page_content for doc in docs]
        return "\n".join(text_list)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
#-----------------

#load tools-----------------

from langchain.agents import load_tools
from langchain.llms import OpenAI
from langchain.agents import initialize_agent

#-----------
from langchain.agents import AgentType

tools = [CustomSearchTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
llm=OpenAI(model_name="text-davinci-003", temperature=0.7, openai_api_key=API_KEY)

agent.run("write a slide for presentation about understanding of GEM Engserv's Methodology for delivering required services to QCI. This content would be a 1 slide pitch")

#q: how to call the CustomSearchTool in below tool names?
tool_names = ["llm-math"]

tools = load_tools(tool_names, llm=llm)

agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)
agent.run("what is salary of brand manager in mumbai. tell for shashikant. multiple it by 4 and return a final figure in INR. Then, write a 1 line summary of headlien for him that has salary value.")

#agent.run("who is latest president of india?")
#-----------------


#initialise agent-----------------

from langchain.agents import Tool
tool_list = [
    # Tool(
    #     name = "llm-math",
    #     func=tools[0].run,
    #     description="Useful for when you need to answer questions about math"
    # ),
    # Tool(
    #     name= "serpapi",
    #     func=tools[0].run,
    #     description= "A search engine. Useful for when you need to answer questions about current events"
    # ),
    Tool(
        name= "presenter",
        func=tools[0].run,
        description= "useful tool for writing powerful presentations"
    ),    
]

from langchain.agents import initialize_agent

agent = initialize_agent(tool_list, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)
agent.run("write a slide for presentation about understanding of GEM Engserv's scope for QCI. This content would be a 1 slide pitch")

#-----------------

#run agent-----------------

agent.run("What is 100 devided by 25?")

#-----------------

#custom tools-----------------

from typing import Optional
from langchain.tools import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class CustomSearchTool(BaseTool):
    name = "personality search tool"
    description = "useful for when you want to search for a personality"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        store = vectorstore.as_retriever()
        docs = store.get_relevant_documents(query)
        text_list = [doc.page_content for doc in docs]
        return "\n".join(text_list)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
#-----------------

#run custom tool-----------------

from langchain.agents import AgentType

tools = [CustomSearchTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("about Shashikant's personality")

#-----------------



