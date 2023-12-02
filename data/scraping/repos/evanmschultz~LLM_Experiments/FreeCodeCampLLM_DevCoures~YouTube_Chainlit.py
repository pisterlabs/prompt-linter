from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.tools import YouTubeSearchTool

import os
import chainlit
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os


@chainlit.on_chat_start
def start():
    tool = YouTubeSearchTool()

    tools: list[Tool] = [
        Tool(
            name="Search",
            func=tool.run,
            description="Useful for when you need to give links to youtube videos. Remember to put https://youtube.com/ in front of every link to complete it",
        )
    ]

    agent: AgentExecutor = initialize_agent(
        tools=tools,
        llm=OpenAI(temperature=0),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    chainlit.user_session.set("agent", agent)


@chainlit.on_message
async def main(message):
    agent: AgentExecutor = chainlit.user_session.get("agent")  # type: ignore
    callbacks = chainlit.LangchainCallbackHandler(stream_final_answer=True)

    await chainlit.make_async(agent.run)(message, callbacks=[callbacks])
