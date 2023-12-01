from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from random_number_tool import random_number_tool
from youTube_helper import youtube_tool
from url_scraping_tool import url_scraping_tool
from current_time_tool import current_time_tool
from wiki_tool import wiki_tool
from weather_tool import weather_tool
from sqldb import sql_tool
from arxiv_tool import arxiv_doc_tool
from gutenburg_tool import gutenberg_doc_tool

from langchain.llms import OpenAI

# define llm
llm = OpenAI(temperature=0.1)

tool_names = [
    "serpapi",  # for google search
    "llm-math",  # this particular tool needs an llm too, so need to pass that
]
tools = load_tools(tool_names=tool_names, llm=llm)
tools.append(youtube_tool)
tools.append(random_number_tool)
tools.append(url_scraping_tool)
tools.append(random_number_tool)
tools.append(current_time_tool)
tools.append(wiki_tool)
tools.append(arxiv_doc_tool)
tools.append(gutenberg_doc_tool)
tools.append(weather_tool)

# create embeddings for the tool retrieval, depending on the query the get_tools will pick the appropriate tool.
docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools)]
model_name = "hkunlp/instructor-xl"
vector_store = Chroma.from_documents(docs, HuggingFaceInstructEmbeddings(model_name=model_name))

retriever = vector_store.as_retriever()

def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    print("relevant docs --> ", docs)
    return [tools[d.metadata["index"]] for d in docs]