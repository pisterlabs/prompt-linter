# Import necessary modules from Langchain library
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
import chainlit as cl
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os

# Adding OPENAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = "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"

@cl.on_chat_start
def start():
    agent_executor = create_python_agent(
    llm=OpenAI(temperature=0.5, max_tokens=2000),  # Initialize the OpenAI language model
    tool=PythonREPLTool(),  # Use a Python REPL tool
    verbose=True,  # Enable verbose mode for debugging
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Specify the agent type
)

    # Store the agent in the user session
    cl.user_session.set("agent",agent_executor)

@cl.on_message
async def main(message):
    # Retrieve the agent from the user session
    agent = cl.user_session.get("agent")

    # Create a callback handler for Langchain
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    # Run the agent with the user's message and the callback handler
    await cl.make_async(agent.run)(message, callbacks=[cb])
