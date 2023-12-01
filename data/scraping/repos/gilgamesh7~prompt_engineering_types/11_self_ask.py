import os
import openai
import dotenv

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for searching"
    )
]


self_ask_with_search = initialize_agent(tools, llm, agent="self-ask-with-search", verbose=True)
self_ask_with_search.run("Who was president of the U.S. when superconductivity was discovered?")
