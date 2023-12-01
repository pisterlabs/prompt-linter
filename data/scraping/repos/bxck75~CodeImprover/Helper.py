import os
import sys
import textwrap
import autopep8
import coverage
from pathlib import Path
from g4f import models
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from langchain import HuggingFaceHub, LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain_experimental.autonomous_agents import HuggingGPT
from langchain.chains.summarize import load_summarize_chain

# Add the path to the directory containing your_script.py
script_dir = os.path.dirname(os.path.abspath(__file__))
two_folders_up = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(two_folders_up)

# Load environment variables
load_dotenv()


# Initialize LLM
llm: LLM = G4FLLM(
    model=models.gpt_35_turbo,
)
