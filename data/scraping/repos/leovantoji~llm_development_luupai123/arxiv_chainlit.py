import chainlit as cl
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI


@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.3, streaming=True)
    tools = load_tools(["arxiv"])

    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        max_iterations=5,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    cl.user_session.set(key="agent_chain", value=agent_chain)


@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent_chain")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(message, callbacks=[cb])
