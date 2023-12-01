import json
from task_manager import TasksManager
from langchain.agents import *
from langchain.utilities.markdown_conversions import convert_to_markdown
from langchain.utilities import google_serper
from langchain.utilities import requests
from langchain.utilities import duckduckgo_search
from langchain.utilities import wikipedia
from langchain.utilities import python
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain.chains import MultiRouteChain


class Multiagent:
    """
    This class uses the different agents by using the
    MultiRouteChain class
    Capable of using different agents, for many tasks, focused on:
    - code generation
    - refactoring
    - git commands
    - file and folder access
    - file and folder creation
    - file and folder deletion
    - file and folder renaming
    - reading and writing file contents
    - reading and writing python code
    - reading and writing text
    - reading and writing markdown
    - reading and writing json
    - reading and writing yaml
    - executing python code
    - executing shell commands
    - executing git commands
    - searching the internet
    - interacting with the user
    - scraping a web page
    - summarizing text
    - converting text to markdown
    - converting text to json
    - converting text to yaml
    - parsing responses
    - calling external APIs
    - checking in with task manager and other agents
    - dynamically adjusting it's prompt templates
    - creating custom agents dynamically..
    - much more!
    """
    def __init__(self, agent_name, task_manager, **kwargs):
        self.agent_name   = agent_name
        self.task_manager : TasksManager = task_manager
    # Implement the rest of the functionalities here...