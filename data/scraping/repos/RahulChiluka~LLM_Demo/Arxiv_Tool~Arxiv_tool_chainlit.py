from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os
import chainlit as cl

os.environ["OPENAI_API_KEY"] = "ADD_YOUR_API_KEY_HERE"
openai.api_key = "ADD_YOUR_API_KEY_HERE"

@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    tools = load_tools(
        ["arxiv"]
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=5, ##Number of papers to search
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
        verbose=True, ##Show the process along with the final result in the terminal
        handle_parsing_errors=True, ##Handles errors using builtin langchain tools
    )

    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])