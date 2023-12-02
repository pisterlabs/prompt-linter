# https://python.langchain.com/en/latest/use_cases/autonomous_agents/autogpt.html
import sys

sys.path.append(r"../")
# from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    verbose=True
)

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

# 使用搜索工具、写入文件工具和读取文件工具设置 AutoGPT
search = SerpAPIWrapper()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# 设置内存
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# 设置模型和 AutoGPT
from langchain.experimental import AutoGPT

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever()
)
# Set verbose to be true
agent.chain.verbose = True

# Run an example
# agent.run(["请用中文写一份今天的A股股市报告"])
agent.run(["请用中文写一份关于《利用 ChatGPT 进行相似匹配》的调研报告，最好包含实现代码"])
