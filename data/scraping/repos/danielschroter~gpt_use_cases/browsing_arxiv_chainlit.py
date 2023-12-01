
import chainlit as cl
from langchain.chat_models import ChatOpenAI

from langchain.agents import load_tools, initialize_agent, AgentType

import os
import openai
import dotenv

dotenv.load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")


@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)

    tools = load_tools(["arxiv", "pubmed"])

    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    cl.user_session.set("agent", agent_chain)



@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(message, callbacks=[cb])