from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.docstore import InMemoryDocstore
###

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.tools import DuckDuckGoSearchRun

import os
import dotenv
import faiss

dotenv.load_dotenv()

api_key = os.getenv("API_KEY")
llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0)

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]


embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="Mr. Researcher",
    ai_role="Researcher",
    memory=vectorstore.as_retriever(),
    tools=tools,
    llm=llm,
)



agent.run("""Make a research using search tool. How to evaluate virtual assistant LLMs? which benchmarks are they using?
""")

# agent.run("""I want to research how many evaluation
#           frameworks are there for virtual_assistant LLMs? which datasets are they using?
#           answer please in json format:"frameworks": {
#                         "lm_eval": {
#                             "description": "lm_eval is a framework...",
#                             "datasets": {"HellaSwag": {
#                                 "description": "HellaSwag is a new video-and-language inference dataset..."
#                                                     }
#                             }
#                     }""")
