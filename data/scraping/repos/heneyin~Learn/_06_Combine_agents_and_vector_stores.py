"""
https://python.langchain.com/docs/modules/agents/how_to/agent_vectorstore

combine agents and vectorstores

 ingested your data into a vectorstore and  want to interact with it in an agentic manner.

创建一个可检索QA

你可以使用多个不同的 vectordbs，并且使用 agent 去选择对应的 vectordbs，有两种方式：
1. 让使用 向量存储的 agent 作为一个普通的工具。
2. 设置 return_direct=True 去直接使用 agent 去路由。
"""

"""
Create the Vectorstore
"""
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
doc_path = str(Path(*relevant_parts) / "state_of_the_union.txt")

from langchain.document_loaders import TextLoader

loader = TextLoader(doc_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")

state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")


docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
)

"""
Create the Agent
"""

# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)


agent.run("Why use ruff over flake8?")


"""
Use the Agent solely as a router
"""
tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
        return_direct=True,
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
        return_direct=True,
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)

agent.run("Why use ruff over flake8?")

"""
Multi-Hop vectorstore reasoning
"""

tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "What tool does ruff use to run over Jupyter Notebooks? Did the president mention that tool in the state of the union?"
)