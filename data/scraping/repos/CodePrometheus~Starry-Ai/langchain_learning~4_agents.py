from langchain.llms import OpenAI

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)
# chat = ChatOpenAI(
#     temperature=0,
#     verbose=True
# )

"""
实现类似 autogpt 的效果
The Python Agent is designed to write and execute Python code to answer a question. 
"""

from langchain.python import PythonREPL

python_repl = PythonREPL()
python_repl.run("print(1+1)")

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)
# agent_executor.run("What is the 10th fibonacci number?")
# agent_executor.run("爬取百度首页源码")
agent_executor.run("针对 A 股进行量化交易，帮我赚钱")

"""
实现带搜索功能的 ChatGPT 
SERP API + Wolfram Alpha
"""

# pip install wolframalpha google-search-results
from langchain.agents import load_tools
from langchain.agents import initialize_agent

tool_names = ["serpapi", "wolfram-alpha"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("What is LangChain?")
agent.run("who is the ceo of pipe?")
agent.run("What is the asthenosphere?")

"""
实现跟 csv 文件进行问答
"""

from langchain.agents import create_csv_agent

agent = create_csv_agent(llm, 'data/know.csv', verbose=True)
agent.run("how many rows are there?")

"""
实现通过自然语言执行SQL命令
"""

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

# mysql+pymysql://user:pass@some_mysql_db_address/db_name
db = SQLDatabase.from_uri(
    "mysql+pymysql://marking-util-dev_ddl:h3tZ90xt2yV(beDU@10.31.117.178:3306/marking-util-dev?charset=utf8",
    include_tables=['oc_knowledge_management'],
    sample_rows_in_table_info=2
)
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
agent_executor.run("Describe the oc_knowledge_management table")
agent_executor.run("获取1条跟发票相关的知识")

"""
实现从矢量存储中检索信息 create_vectorstore_agent
"""

from langchain.document_loaders import DirectoryLoader

# 加载文件夹中的所有txt类型的文件，并转成 document 对象
loader = DirectoryLoader('./data/', glob='**/*.txt')
documents = loader.load()
# 接下来，我们将文档拆分成块。
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# 然后我们将选择我们想要使用的嵌入。
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
# 我们现在创建 vectorstore 用作索引，并进行持久化
from langchain.vectorstores import Chroma

state_of_union_store = Chroma.from_documents(texts, embeddings, collection_name="knowledge")

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

vectorstore_info = VectorStoreInfo(
    name="knowledge",
    description="primary question, similar question, and other",
    vectorstore=state_of_union_store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
agent_executor.run("knowledge: 出差申请单修改. Answer: ")
