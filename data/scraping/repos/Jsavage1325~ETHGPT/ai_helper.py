"""
This is an AI helper which will query the docs of required libraries and attempt to write valid code using them
"""
from retrieve_docs import LangchainDocSearch, AirStackDocSearch
from CodeWriter import PythonCodeWriter
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools
from langchain.tools import Tool
from langchain import SerpAPIWrapper
from langchain.tools import tool

llm = OpenAI(temperature=0.0)

search = SerpAPIWrapper()

tools = [
    # Tool(
    #     name="Intermediate Answer",
    #     func=search.run,
    #     description="Useful for when you need to search langchain docs"
    # ),
    LangchainDocSearch(),
    PythonCodeWriter(),
    # AirStackDocSearch()
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.agent.llm_chain.prompt.template)
agent.run({'input': "Can you analyse the smart contract code ", 'chat_history': []})


"""
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

llm = OpenAI(temperature=0)

from pathlib import Path
relevant_parts = []
for p in Path(".").absolute().parts:
    relevant_parts.append(p)
    if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
        break
doc_path = str(Path(*relevant_parts) / "some_cool_data.csv")

from langchain.document_loaders import TextLoader
loader = TextLoader(doc_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_spl
"""