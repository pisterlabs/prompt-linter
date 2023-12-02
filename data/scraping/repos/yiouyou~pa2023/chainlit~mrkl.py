from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl

from dotenv import load_dotenv
load_dotenv()


@cl.on_chat_start
def start():
    llm = OpenAI(temperature=0, streaming=True)
    google_serper = GoogleSerperAPIWrapper()
    llm_chat = ChatOpenAI(temperature=0, streaming=True)
    llm_chat_math = LLMMathChain.from_llm(llm=llm_chat, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=google_serper.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
            func=llm_chat_math.run,
            description="useful for when you need to answer questions about math",
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=True
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(message, callbacks=[cb])

