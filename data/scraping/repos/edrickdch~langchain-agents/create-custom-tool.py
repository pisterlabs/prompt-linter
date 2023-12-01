from langchain import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

# Initialize the LLM to use for the agent
llm = ChatOpenAI(temperature=0)

# Load the tool configs that are needed.
search = SerpAPIWrapper()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need \
            to answer questions about current events",
    ),
]
