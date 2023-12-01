import os
import json
from tkinter import filedialog
from docx import Document
from typing import Optional, List, Dict, Callable, Union 
import autogen
from autogen import AssistantAgent, ConversableAgent, UserProxyAgent, ChatCompletion
from langchain.document_loaders import PyPDF2Loader, AsyncHtmlLoader, JSONLoader, TextLoader, Docx2txtLoader
from src.prompts import SPR_GENERATOR_SYS_MSG, SPR_INTERPRETER_SYS_MSG

# TODO: add methods to add text to the SPR 
# TODO: add methods to load conversation history



gpt3 = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "model": "gpt-3.5-turbo-16k",
    "temperature": 0,
    "request_timeout": 300,
}
gpt4 = {
    "api_key": os.environ["OPENAI_API_KEY"],
    "model": "gpt-4",
    "temperature": 0,
    "request_timeout": 500,
}

config_list = [
        {
            "model": "gpt-4",
            "api_key": os.environ["OPENAI_API_KEY"],
            "temperature": 0,
            "request_timeout": 300,
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "open_ai",
            "api_base": "https://api.openai.com/v1",
        },
       {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "open_ai",
            "api_base": "https://api.openai.com/v1",
        }
       ]

class SprGeneratorAgent(ConversableAgent):
    def __init__(
        self,
        name: str = "Sparse_Priming_Representation_Generator",
        system_message: str = SPR_GENERATOR_SYS_MSG,
        human_input_mode: Optional[str] = "NEVER",
        llm_config: Dict = Optional[gpt3],
        max_consecutive_auto_reply: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(
            llm_config,
            max_consecutive_auto_reply,
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            **kwargs
            )
        

class SprInterpreterAgent(ConversableAgent):
    def __init__(
        self,
        name: str = "Sparse_Priming_Representation_Interpreter",
        system_message: str = SPR_INTERPRETER_SYS_MSG,
        human_input_mode: Optional[str] = "NEVER",
        llm_config: Dict = gpt3,
        max_consecutive_auto_reply: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(
            llm_config,
            max_consecutive_auto_reply,
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            **kwargs
            )

class MemoryFileLoader:
    """Functions as a wrapper for document loaders from langchain."""
    def __init__(self, docx_loader, txt_loader, json_loader):
        self.docx_loader = docx_loader
        self.txt_loader = txt_loader 
        self.json_loader = json_loader
    
    def choose_file(self):
        """Opens a file dialog to choose a file to load."""
        file_types = [
            ("Word Documents", "*.docx"),
            ("Text Documents", "*.txt"),
            ("JSON Files", "*.json"),
            ]
        saved_doc = filedialog.askopenfilename(
                title="Select a file to load",
                filetypes=file_types,
                defaultextension=".txt",
            )
        return saved_doc
    
    def load_docx(self, saved_doc: str):
        """Loads a docx file."""
        self.docx_loader = Docx2txtLoader()
        document = self.docx_loader.load(saved_doc)
        return document

context = {"role": "system","content": SPR_GENERATOR_SYS_MSG  }
        
def spr_compress(prompt: str, document: str, llm_config: Dict = gpt3, **kwargs):
    """Compresses a document into an SPR.""" 
    spr_agent = SprGeneratorAgent(llm_config=llm_config)
    user = autogen.UserProxyAgent(
        name="USER",
        human_input_mode="NEVER",
        code_execution_config={},
    )
    
     


        
               

    
    
    
