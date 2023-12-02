import os
from apikey import serpapikey, openaikey, wolframalphaapikey, googleapikey, googlesearch

os.environ['SERPAPI_API_KEY'] = serpapikey
os.environ['OPENAI_API_KEY'] = openaikey
os.environ['WOLFRAM_ALPHA_APPID'] = wolframalphaapikey
os.environ['GOOGLE_API_KEY'] = googleapikey
os.environ['GOOGLE_CSE_ID'] = googlesearch


from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.agents import Tool

from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss

# Define your tools
wolframalpha = WolframAlphaAPIWrapper()
googlesearch = GoogleSearchAPIWrapper()
arxiv = ArxivAPIWrapper()

tools = [
    Tool(
        name="googlesearch",
        func=googlesearch.run,
        description="Useful for searching the web for information."
    ),
    Tool(
        name="arxiv",
        func=arxiv.run,
        description="Useful for when you need scientific research papers."
    ),
    Tool(
        name="wolframalpha",
        func=wolframalpha.run,
        description="Useful for answering factual questions."
    ),
    WriteFileTool(),
    ReadFileTool()
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()

# Initialize the vectorstore as empty
embedding_size = 1536  # openai embeddings has 1536 dimensions
index = faiss.IndexFlatL2(embedding_size)  # Index that stores the full vectors and performs exhaustive search
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# Initialize the agent
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI

agent = AutoGPT.from_llm_and_tools(
    ai_name="LENR_AutoGPT",
    ai_role="LENR Research Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0.7),
    memory=vectorstore.as_retriever()
)

# Set verbose to be true
agent.chain.verbose = True

agent.run([("Goal: Research and Develop a Mathematical Theory for LENR.")])