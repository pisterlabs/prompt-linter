import openai
import numpy as np
import os
from dotenv import load_dotenv
import os
from collections import deque
from typing import Dict, List, Optional, Any
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
# Langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain ,LLMCheckerChain
from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    StringPromptTemplate
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from typing import Optional
from langchain.chains import SimpleSequentialChain ,SequentialChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent,BaseMultiActionAgent
from langchain.agents import AgentType, initialize_agent,AgentExecutor,BaseSingleActionAgent
from langchain.tools import tool
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser,Agent
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
import jsonschema
from jsonschema import ValidationError
import logging

from .chains import (
    questionClassifier,
    questionGurobiClassifier
)

from .utils import num_tokens_from_string, extract_json_schema, Visualizer
import logging

from typing import Tuple, Union
from plotly import graph_objects as go
import time

class outputValidator:
    """
    Validates json output of any llmchain, given that it's created with the create_structured_output_chain function
    Gives Feedback to the LLMChain to improve the output
    """
    @staticmethod
    def _getOutputSchemaMapping(LLMChain:LLMChain) -> dict:
        """Returns the supposed output schema of the given LLMChain 
        
        Example llm_kwargs:
        {'functions': [{'name': 'output_formatter',
   'description': 'Output formatter. Should always be used to format your response to the user.',
   'parameters': {'name': 'binary_classifier_article_schema',
    'description': 'Binary Classifier Schema for Article, 0 for False and 1 for True',
    'type': 'object',
    'properties': {'isDisruptionEvent': {'type': 'boolean'},
     'Reason': {'type': 'string'}},
    'required': ['isDisruptionEvent', 'Reason']}}],
    'function_call': {'name': 'output_formatter'}}
    """
        output_schema = LLMChain.llm_kwargs
        # Create a dictionary mapping with Key as Key and Value as type
        validator_json_schema = extract_json_schema(output_schema)
        return validator_json_schema
        
    @classmethod
    def validate(cls, LLMChain:LLMChain, output: dict) -> Tuple[bool, str]:
        """Validates the output of the given llmchain
        Args:
            JSON_SCHEMA (dict): The JSON Schema to validate
            output (dict): The output of the LLMChain
        Returns:
            Tuple[bool, str]: A tuple containing a boolean and a string. 
            The boolean is True if the output is valid according to the schema, False if not. 
            The string is the reason for the boolean value. If the boolean is True, the string will be empty.
            The string might be used to give feedback to LLMChain to improve the output.
        """
        # Check if the output is a dict
        if not isinstance(output,dict):
            return (False,"The output is not JSON object (dict)")
        # Get the output schema mapping
        JSON_SCHEMA = cls._getOutputSchemaMapping(LLMChain)
        
        try:
            # Create a validator based on the JSON_SCHEMA
            validator = jsonschema.Draft7Validator(JSON_SCHEMA)
            
            # Check if the output is valid
            errors = list(validator.iter_errors(output))
            
            if errors:
                error_messages = [str(error) for error in errors]
                return (False, ", ".join(error_messages))  # Output is invalid, return the validation error messages
            else:
                return (True, "")  # Output is valid
        except Exception as e:
            return (False, str(e))  # Handle any other exceptions that may occur

class AgentMain:
    """
    Main Agent that handle all the Gurobi Optiguide and Visualization Query Database tools
    """

    questionClassifierChain:LLMChain = questionClassifier
    questionGurobiClassifierChain:LLMChain = questionGurobiClassifier

    @classmethod
    def ask(cls,question:str) -> Union[str,go.Figure]:
        # Check if the article is a disruption event article
        classifier_result = cls._binaryClassifier(question)

        if not classifier_result['isRelevant']:
            print(f''':red[NOT A RELEVANT QUESTION] \nREASONING:**"{classifier_result["Reason"]}"**\nPlease ask another question''')
            return f"Not a relevant question {question},\nREASONING:{classifier_result['Reason']}\nPlease ask another question"
        print(f''':green[Relevant Questions] *"{question}"*
              REASONING:**{classifier_result["Reason"]}**''')
        
        gurobiQuestion_result = cls._gurobiClassifier(question)

        if not gurobiQuestion_result['isRelevant']:
            # This means is a visualization question
            print(f':blue[Requires data from database]')
            print(f''':green[Querying Database...]''')
            # Set timeout 2 seconds
            time.sleep(2)
            print(f''':green[Retrived Data: showing first 5rows:]\n\n ```python
                  {Visualizer.getWarehouseData().head()}```''')
            print(f''':blue[Generating Code for Visualization...]''')
            time.sleep(2)
            print(f''':green[Code Generated:]\n\n ```py{Visualizer.get_code()}```''')
            print(f''':green[Sucessfully Generated plot]''')
            return Visualizer.plot_current_warehouse_capacity()
        
        else:
        # This means is a gurobi question
            print(f':blue[Running LLM Agent (Will take 30~ sec)...]')
            return question
    
    
    @staticmethod
    def _binaryClassifier(question: str) -> dict:
        """Classifies if the given article is a valid disruption event article or not
        Always use summary instead of full article for token saving

        Returns:
            dict: A dictionary containing the classification result and the reason for the classification result
            Keys:
                isDisruptionEvent: bool
                Reason: str
                disruptionType: str

        Raises Exception if the binary classification fails
        """
        try:
            # Get number of tokens for the article
            result = AgentMain.questionClassifierChain.run(question=question, feedback="")

            # Validate the output
            validation_results = outputValidator.validate(AgentMain.questionClassifierChain, result)
            if not validation_results[0]:
                print(f'_binaryClassifier Validation Error, Re-Running with feedback: {validation_results[1]}')
                # Re-run the validation error as feedback for the LLMChain
                result = AgentMain.questionClassifierChain.run(question=question, feedback=validation_results[1])

            return result
        except Exception as e:
            raise Exception(f"Error: Binary Classification Failed -> {e}") from e
    
    @staticmethod
    def _gurobiClassifier(question:str) -> dict:
        """Classifies if the given article require use of gurobi optimization or not"""
        try:
            # Get number of tokens for the article
            result = AgentMain.questionGurobiClassifierChain.run(question = question, feedback="")

            # Validate the output
            validation_results = outputValidator.validate(AgentMain.questionGurobiClassifierChain, result)
            if not validation_results[0]:
                print(f'_binaryClassifier Validation Error, Re-Running with feedback: {validation_results[1]}')
                # Re-run the validation error as feedback for the LLMChain
                result = AgentMain.questionGurobiClassifierChain.run(question=question, feedback=validation_results[1])

            return result
        except Exception as e:
            raise Exception(f"Error: Binary Classification Failed -> {e}") from e        

    @staticmethod
    def visualize() -> go.Figure:
        """
        Method to visuzalize the current warehouse capacity 
        """
        pass
