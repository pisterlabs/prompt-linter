# Import necessary modules from Langchain and Chainlit libraries
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os

# Adding OPENAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = "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"

# Define a function to run when the chat session starts
@cl.on_chat_start
def start():
    # Initialize ChatOpenAI with specific settings
    llm = ChatOpenAI(temperature=0, streaming=True)

    # Load tools, in this case, only "arxiv"
    tools = load_tools(["arxiv"])

    # Initialize an agent with the specified tools and settings
    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # Store the agent in the user session
    cl.user_session.set("agent", agent_chain)

# Define an async function to handle user messages
@cl.on_message
async def main(message):
    # Retrieve the agent from the user session
    agent = cl.user_session.get("agent")

    # Create a callback handler for Langchain
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    # Run the agent with the user's message and the callback handler
    await cl.make_async(agent.run)(message, callbacks=[cb])
