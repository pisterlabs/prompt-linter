import chainlit as cl

from sql_analyzer.agent_factory import agent_factory
from langchain.agents import AgentExecutor


@cl.on_chat_start
def start():
    agent_executor = agent_factory()
    cl.user_session.set("agent", agent_executor)


@cl.on_message
async def main(message):
    agent_executor: AgentExecutor = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    resp = await cl.make_async(agent_executor.run)(message, callbacks=[cb])
    final_message = cl.Message(content=resp)
    await final_message.send()
