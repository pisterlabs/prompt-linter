import chainlit as cl
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os
from config import open_ai_key, serp_api_key

# Set API Keys
os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['SERPAPI_API_KEY'] = serp_api_key

# Initialize the Agent
agent_executor = create_python_agent(
    llm=OpenAI(temperature=0.5, max_tokens=2000),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

@cl.on_chat_start
def start():
    # Any initialization code goes here
    pass

@cl.on_message
async def main(message):
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent_executor.run)(message, callbacks=[cb])
