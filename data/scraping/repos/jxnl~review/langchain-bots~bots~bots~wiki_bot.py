from dotenv import load_dotenv

load_dotenv()

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.base import DocstoreExplorer

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(name="Search", func=docstore.search),
    Tool(name="Lookup", func=docstore.lookup),
]

llm = OpenAI(temperature=0, model_name="text-davinci-003")
react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)

if __name__ == "__main__":
    result = react.run("Who is the president of the United States?")
