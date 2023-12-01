from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatAnthropic
import asyncio
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)

from dotenv import load_dotenv
load_dotenv()

import nest_asyncio

nest_asyncio.apply()


async def go(async_browser):
    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()

    llm = ChatAnthropic(temperature=0)  # or any other LLM, e.g., ChatOpenAI(), OpenAI()

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )


    response = await agent_chain.arun(input="Browse to 2023.teemconference.eu and tell me when is the new deadline for submissions, please.")
    print(response)
    



async def main():
 async with create_async_playwright_browser() as playwright:
        await go(playwright)

asyncio.run(main())