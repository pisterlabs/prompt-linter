from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.agents import tool
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import textwrap
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
import langchain
from slack_sdk import WebClient
import json
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.schema import Document
from typing import Iterator, List, Literal, Optional, Sequence, Union
from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain.document_loaders.blob_loaders import Blob
import os
import openai

from audio.audio_extractor import AudioExtractor

class SimpleAudioInfoExtractor(AudioExtractor):
    def __init__(self) -> None:
        pass

    def audio_to_text(self, urls,save_dir):
        #  Moive Clip
        #urls = ["https://www.youtube.com/watch?v=5Ay5GqJwHF8&list=PL86SiVwkw_oc8r_X6PL6VqcQ7FTX4923M&index=1"]
        # Directory to save audio files
        #save_dir = "/Users/bpakra200/Downloads/YouTubeclip"
        try:
            # Transcribe the videos to text
            loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())

            docs = loader.load()
            print("After load")
            if len(docs) == 0:
                print("No documents loaded.")
            else:
                for doc in docs:
                # Process the loaded documents
                    print(doc)
                return docs 
        except Exception as e:
            print("Error loading documents:", str(e))   

