import os
os.environ["LANGCHAIN_TRACING"] = "true" # If you want to trace the execution of the program, set to "true"

'''
from the command line, run langchain-server
'''

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser, # A synchronous browser is available, though it isn't compatible with jupyter.
)

import asyncio

# This import is required only for jupyter notebooks, since they have their own eventloop
import nest_asyncio
nest_asyncio.apply()

async def go(async_browser):
    # async_browser = 
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = browser_toolkit.get_tools()
    
    # print(tools)

    llm = ChatOpenAI(temperature=0) # Also works well with Anthropic models
    agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


    response = await agent_chain.arun(input="Browse to 2023.teemconference.eu and tell me when is the new deadline for submissions, please.")
    print(response)

    response = await agent_chain.arun(input="What's the latest xkcd comic URL? (I need just the png URL, please)")
    print(response)




async def main():
 async with create_async_playwright_browser() as playwright:
        await go(playwright)

asyncio.run(main())