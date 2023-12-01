from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl

os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""


@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0, streaming=True)
    llm1 = OpenAI(temperature=0, streaming=True)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]
    agent = initialize_agent(
        tools, llm1, agent="chat-zero-shot-react-description", verbose=True
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])
