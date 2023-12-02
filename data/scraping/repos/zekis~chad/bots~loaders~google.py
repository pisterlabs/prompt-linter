import traceback
import config

from dotenv import find_dotenv, load_dotenv
#from flask import request
import json
import os
import re
import pika
import faiss
import urllib
from urllib.parse import quote

from pydantic import BaseModel, Field
from datetime import datetime, date, time, timezone, timedelta
from typing import Any, Dict, Optional, Type

from bots.loaders.todo import MSGetTasks, MSGetTaskFolders, MSGetTaskDetail, MSSetTaskComplete, MSCreateTask, MSDeleteTask, MSCreateTaskFolder
from bots.rabbit_handler import RabbitHandler
#from bots.utils import encode_message, decode_message, generate_response, validate_response, parse_input, sanitize_string
from common.rabbit_comms import publish, publish_list, publish_draft_card, publish_draft_forward_card
from common.utils import tool_description, tool_error
#from common.utils import generate_response, generate_whatif_response, generate_plan_response
#from bots.langchain_assistant import generate_response
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool
from langchain.tools import StructuredTool

#from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate
from langchain.agents import load_tools, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, AgentType

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader



class GoogleBot(BaseTool):
    parameters = []
    optional_parameters = []
    name = "GOOGLE"
    summary = """useful for when you want to search the internet using google. """
    parameters.append({"name": "query", "description": "search query" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    def _run(self, query: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            website = "https://www.google.com.au"
            # if text:
            #     #question = text.get("question")
            #     website = text.get("website")
            
            #URL = urllib.parse.quote(website)
            print(f"{website} -> {query}")
            url_encoded_s = quote(query)

            llm = ChatOpenAI(temperature=0)
           
            
            # text_splitter = RecursiveCharacterTextSplitter(
            #     # Set a really small chunk size, just to show.
            #     chunk_size = 100,
            #     chunk_overlap  = 20,
            #     length_function = len,
            # )
            hwebsite = ensure_http_or_https(website)
            loader = WebBaseLoader(hwebsite + "/search?q=" + query)
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            splitted_documents  = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()

            print(f"Web texts: {documents}")
            
            web_db = Chroma.from_documents(splitted_documents, embeddings, collection_name="web")
            #web_db.persist()

            
            #chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=web_db.as_retriever())
            #response = chain({"question": query}, return_only_outputs=True)
            #publish(response)
            response = web_db.similarity_search(query)

            if publish.lower() == "true":
                publish(response[0].page_content)
                return config.PROMPT_PUBLISH_TRUE
            else:
                return response[0].page_content
            
        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)
        

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BROWSE does not support async")

def ensure_http_or_https(url):
    if not url.startswith('http://') and not url.startswith('https://'):
        # Add https:// if the URL does not start with http:// or https://
        return 'https://' + url
    # If it already starts with http:// or https://, return it as is
    return url