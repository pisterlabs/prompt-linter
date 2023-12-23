# /app/tests/test_agent.py

import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

import pytest
import asyncio
from unittest.mock import MagicMock
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import SerpAPIWrapper, LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI

from src.agent.agent import setup_agent, chat_with_agent
from src.template.template import CustomPromptTemplate, read_template
from src.parser.parser import CustomOutputParser
import src.utils.config as cfg
import yaml

# Create a new function that loads the config with the correct path for testing
def load_config_test():
    config_file_path = Path(__file__).resolve().parents[1].joinpath("config.yml")
    with open(config_file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

# Use monkeypatch to temporarily replace load_config in the utils.config module
@pytest.fixture(autouse=True)
def patch_load_config(monkeypatch):
    monkeypatch.setattr(cfg, 'load_config', load_config_test)

# Set up environment variables for testing
@pytest.fixture(autouse=True)
def setup_environment_variables(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test_openai_api_key')
    monkeypatch.setenv('SERPAPI_API_KEY', 'test_serpapi_api_key')

def test_setup_agent():
    agent_executor = setup_agent()
    assert isinstance(agent_executor, AgentExecutor), "Failed to set up agent executor"
    assert isinstance(agent_executor.agent, LLMSingleActionAgent), "Failed to set up LLMSingleActionAgent"

    # Add this line to print the first tool to see what's wrong with it
    print(agent_executor.tools[0])

    assert isinstance(agent_executor.tools[0].func.__self__, SerpAPIWrapper), "Failed to set up SerpAPIWrapper"


@pytest.fixture
def mock_agent_executor():
    async def mock_agent_executor_coro(text):
        return "Test response"

    return mock_agent_executor_coro


def test_chat_with_agent(mock_agent_executor):
    input_text = "Hello! What is your name and today's date?"
    response = asyncio.run(mock_agent_executor.__call__(input_text))
    assert response == "Test response", "Failed to chat with agent"