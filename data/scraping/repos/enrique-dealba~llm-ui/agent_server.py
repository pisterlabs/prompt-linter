import argparse
import config
import logging
import os

from typing import List, Dict, Any, Optional, Union
from apikeys import open_ai_key, hf_key, serpapi_key
from prompts import server_template, agent_template #, db_template
from agent_utils import get_agent_thoughts
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR

## AGENT IMPORTS ##
from langchain.agents import load_tools, initialize_agent
from langchain.agents import (
    AgentType,
)  # We will be using the type: ZERO_SHOT_REACT_DESCRIPTION which is standard


os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['HUGGINGFACEHUB_APU_TOKEN'] = hf_key
os.environ['SERPAPI_API_KEY'] = serpapi_key

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

class ListHandler(logging.Handler): # Inherits from logging.Handler
    def __init__(self):
        super().__init__()
        self.log = []

    def emit(self, record) -> None:
        """Keeps track of verbose outputs from agent LLM in logs."""
        self.log.append(self.format(record))

handler = ListHandler()
logging.getLogger().addHandler(handler)

class FlaskServer:
    """Flask server with LLM operations."""

    def __init__(self, use_local: bool=True, template: str="", init_model: bool=True):
        self.app = app 
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.embedding = None
        self.memory = None
        self.qa_chain = None
        self.tools = None
        self.agent = None
        self.template = template
        self.persist_directory = 'docs/chroma/'
        if init_model:
            self.initialize_model(use_local)

    def set_tools(self, use_default_tools: bool):
        """Setup of tools for agent LLM."""
        assert self.llm is not None

        if use_default_tools:
            self.tools = load_tools(["wikipedia",
                                     "serpapi",
                                     "python_repl",
                                     "terminal"],
                                     llm=self.llm)
            
    def init_agent(self):
        """Initializes LLM agent."""
        if self.tools is None:
            self.set_tools(use_default_tools=True)
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True)

    def initialize_model(self, use_local: bool=True):
        """Initialize LLM either locally or from OpenAI (GPT 3.5)."""
        if use_local:
            directory_path = config.MODEL_DIRECTORY_PATH
            self.tokenizer = AutoTokenizer.from_pretrained(directory_path)
            self.model = AutoModelForCausalLM.from_pretrained(directory_path)
        else:
            self.llm = OpenAI(temperature=0.2)
        
        if self.model:
            self.llm = self.initialize_local_model()
            logging.debug(f"LLM running on {self.model.device}")
        
    def initialize_local_model(self) -> HuggingFacePipeline:
        """Initialize local LLM."""
        local_pipe = pipeline("text-generation",
                               model=self.model,
                               tokenizer=self.tokenizer,
                               max_length=500)
        return HuggingFacePipeline(pipeline=local_pipe)
    
    def task_agent(self, prompt: str) -> Union[tuple, jsonify]:
        """Tasks LLM agent to process a given prompt/task."""
        if not prompt:
            return jsonify({'error': 'No task provided'}), 400
        
        self.init_agent()

        assert self.tools is not None
        assert self.agent is not None

        final_answer = self.agent.run(prompt)

        logs = handler.log # gets verbose logs from agent LLM
        cleaned_logs = get_agent_thoughts(logs) # cleans logs

        response = cleaned_logs + [final_answer]

        return jsonify({'response': response})

    def error_handler(self, e: Exception) -> jsonify:
        """Handle errors during request processing."""
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing the request.'}), 500

@app.route('/', methods=['GET', 'POST'])
def home() -> str:
    """Renders home page for LLM agent."""
    return render_template('agent-index.html')

@app.route('/chat', methods=['POST'])
def chat() -> jsonify:
    """Handles chat requests via LLM agent."""
    
    try:
        prompt = request.json['prompt']

        return server.task_agent(prompt)
    except Exception as e:
        return server.error_handler(e)

def create_parser() -> argparse.ArgumentParser:
    """Creates a command-line argument parser."""
    parser = argparse.ArgumentParser(description="Run the Flask server")
    parser.add_argument('--local', action='store_true', help='Use local server settings')
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    server = FlaskServer(use_local=args.local, template=server_template)
    app.run()