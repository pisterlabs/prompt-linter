from langchain.llms.openai import OpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import logging

from src.config import Config

# Set up logging
logger = logging.getLogger(__name__)



def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Give the agent a list of tools to use
def dummy_function(prompt: str) -> str:
    """
    Narrate the story based on the given prompt.
    """

    logger.info(f"Running dummy_function with prompt: {prompt}")
    return prompt

tools: list[StructuredTool] = [
    StructuredTool.from_function(
        name= "A dummy function", 
        func=dummy_function, 
        description="Generates a NPC based on the given prompt.",
        # args_schema={"prompt": {"type": "string", "minLength": 1, "maxLength": 1000}},
    ),
    StructuredTool.from_function(
        name= "Add two numbers", 
        func=add, 
        description="Adds two numbers together.",
        # args_schema={"a": {"type": "integer"}, "b": {"type": "integer"}},
    ),
]

# Make a memory for the agent to use
memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0, openai_api_key=Config().API_KEY)
agent_chain = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory,
    max_iterations=2,
    )

def run_agent(prompt: str) -> str:
    """Run the agent."""
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    if (len(prompt) < 1) or (len(prompt) > 1000):
        raise ValueError("Prompt must be at least 1 character or less than 1000 characters.")
    
    result = agent_chain.run(prompt)
    logger.info(f"Finished running langchain_function_calling.py, result: {result}")
    return result

