import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
import os
from config import open_ai_key, serp_api_key

os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['SERPAPI_API_KEY'] = serp_api_key

# Initialize the LLM and Tools
llm = ChatOpenAI(temperature=0.5)
math_llm = OpenAI(temperature=0.5)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

# Initialize the Agent
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

@cl.on_chat_start
def start():
    # Any initialization code if needed
    pass

@cl.on_message
async def main(message):
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent_chain.run)(message, callbacks=[cb])
