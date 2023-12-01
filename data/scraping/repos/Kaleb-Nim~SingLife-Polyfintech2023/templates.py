import pinecone
import openai
import numpy as np
import os
from dotenv import load_dotenv

# Langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Optional

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.schema import HumanMessage, AIMessage, ChatMessage

# wandb
import wandb 

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_TO_ENV = os.path.join(parent_directory, '.env')
PATH_TO_TEMPLATES = os.path.join(parent_directory, 'example_templates')
print('PATH_TO_TEMPLATES: ', PATH_TO_TEMPLATES)
print('PATH_TO_ENV: ', PATH_TO_ENV)



class Templates:

    """
    Purpose:
    Manage all the user defines prompt templates for the LLMChain
    """
    def __init__(self, path_to_templates: str = PATH_TO_TEMPLATES):
        """
        Purpose:
        Load all the templates into a dictionary
        """
        self.path_to_templates = path_to_templates
        print('self.path_to_templates: ', self.path_to_templates)
        self.templates = self.load_templates()
        print('self.templates: ', self.templates)
        # Display number of templates loaded
        print(f"Loaded {len(self.templates)} templates", flush=True)

    def load_templates(self):
        """
        Purpose:
        Load all the templates into a dictionary
        """
        templates = {}
        for filename in os.listdir(self.path_to_templates):
            if filename.endswith(".json"):
                with open(os.path.join(self.path_to_templates, filename), "r") as f:
                    templates[filename] = f.read()

        return templates

    def get_function_call_template(self, template_name: str):
        """
        Purpose:
        Get only function_calling template from the dictionary

        Returns:
            template (str): the template string for function_calling
        """
        self.function_call_prefix = "function_calling"
        return self.templates[f"{self.function_call_prefix}_{template_name}.json"]

    def get_template_names(self):
        """
        Purpose:
        Get the template names from the dictionary
        """
        return list(self.templates.keys())

    def get_template_names_with_prefix(self, prefix: str):
        """
        Purpose:
        Get the template names from the dictionary
        """
        return [name for name in self.templates.keys() if name.startswith(prefix)]


# Testing
if __name__ == "__main__":
    templates = Templates()
    print(templates.get_template_names())
    print(templates.get_template_names_with_prefix('function_calling'))
    print(templates.get_function_call_template('base'))
