

import environment

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.experimental import AutoGPT
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

# Define your embedding model
# embedding = OpenAIEmbeddings()
# llm = ChatOpenAI(temperature=0)

# Initialize the vectorstore as empty
import faiss

# embedding_size = 1536   #For chatgpt OpenAI
embedding_size = 768      #For HuggingFace
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embedding.embed_query, index, InMemoryDocstore({}), {})


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True


agent.run(["write a weather report for SF today"])
