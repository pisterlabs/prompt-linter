# import auxillary packages
import re
import requests  # for loading the example source code
import openai
import os
import streamlit as st

# import flaml and autogen
from flaml import autogen
from flaml.autogen.agentchat import Agent, UserProxyAgent
from optiguide.optiguide import OptiGuideAgent # This one we might need to change things up 
from llm.ICL import example_qa

from dotenv import load_dotenv
path_to_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(path_to_root, ".env"))

# Set the OpenAI API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


def get_code(fileName:str) -> str:
    """Gets the code for the PSA Port Operations use case from src/psa_gurobi.py
    Assumes that the file is in the src/ directory

    Args:
        fileName (str): Name of the file, without the .py extension
    
    """
    with open(os.path.join(path_to_root, f"src/{fileName}.py"), "r") as f:
        return f.read()
    
code = get_code("psa_gurobi")
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-4",
            "gpt4",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
        }
    }
)
agent = OptiGuideAgent(name="CargoLingo Advisor Agent",
                  source_code=code,
                   debug_times=1,
                  example_qa=example_qa,
                llm_config={
        "request_timeout": 600,
        "seed": 42,
        "config_list": config_list,
    })

user = UserProxyAgent("user", max_consecutive_auto_reply=0,
                         human_input_mode="NEVER", code_execution_config=False)
