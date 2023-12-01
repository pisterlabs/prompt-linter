import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding


from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,# A synchronous browser is available, though it isn't compatible with jupyter.
)

# This import is required only for jupyter notebooks, since they have their own eventloop
import asyncio
import nest_asyncio
nest_asyncio.apply()

async_browser = create_async_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
print(tools)

tools_by_name = {tool.name: tool for tool in tools}
navigate_tool = tools_by_name["navigate_browser"]
get_elements_tool = tools_by_name["get_elements"]

async def rrr():
    await navigate_tool.arun({"url": "https://youtube.com"})

    # The browser is shared across tools, so the agent can interact in a stateful manner
    element = await get_elements_tool.arun({"selector": ".container__headline", "attributes": ["innerText"]})
    print(f"Elements: {element}")

    # If the agent wants to remember the current webpage, it can use the `current_webpage` tool
    current_webpage = await tools_by_name['current_webpage'].arun({})
    print(f"Current webpage: {current_webpage}")

asyncio.run(rrr())
# asyncio.run(navigate_tool.arun({"url": "https://web.archive.org/web/20230428131116/https://www.cnn.com/world"}))
# asyncio.run(get_elements_tool.arun({"selector": ".container__headline", "attributes": ["innerText"]}))
# asyncio.run(tools_by_name['current_webpage'].arun({}))

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatAnthropic

# llm = ChatAnthropic(temperature=0) # or any other LLM, e.g., ChatOpenAI(), OpenAI()
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_chain = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory, 
    agent_kwargs = {
        "memory_prompts": [chat_history],
        "input_variables": ["input", "agent_scratchpad", "chat_history"]
    }
)
# agent_chain = initialize_agent(tools, 
#                             llm,
#                             agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#                             verbose=True)

async def fff():
    result = await agent_chain.arun("What are the headers on langchain.com?")
    print(result)

# asyncio.run(fff())