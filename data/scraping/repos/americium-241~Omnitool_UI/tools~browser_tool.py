
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import create_sync_playwright_browser
    #create_async_playwright_browser,
 

import sys

"""allow you to navigate using the browser, provide url or keyword and instructions"""    

sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

llm = ChatOpenAI(temperature=0, model="gpt-4")# or any other LLM, e.g., ChatOpenAI(), OpenAI()
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    streaming=True,

)

response = agent_chain.run(sys.argv[1])


