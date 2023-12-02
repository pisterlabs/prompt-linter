from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def DuckDuckGoTool():
    tools = []
    tools.append(Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ))
    return tools

def duckduckgo_search():
    tools = []
    tools.extend(DuckDuckGoTool())
    return tools