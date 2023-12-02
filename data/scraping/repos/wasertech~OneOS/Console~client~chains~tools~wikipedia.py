from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper

encyclopedia = WikipediaAPIWrapper()

def get_tool():
    return [
            Tool(
                name="Wikipedia",
                func=encyclopedia.run,
                description="useful when you need to use an encyclopedia to answer a question; input will be used to search on wikipedia"
                )
        ]