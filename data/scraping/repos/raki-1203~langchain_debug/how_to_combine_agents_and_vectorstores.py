import os

from pathlib import Path

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


if __name__ == '__main__':
    # Create the Vectorstore
    llm = OpenAI(temperature=0)

    relevant_parts = []
    for p in Path('.').absolute().parts:
        relevant_parts.append(p)
        if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
            break

    doc_path = str(Path(*relevant_parts) / "state_of_the_union.txt")
    print(doc_path)

    loader = TextLoader(doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings, collection_name='state-of-union')

    state_of_union = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                                 retriever=docsearch.as_retriever())

    loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")

    docs = loader.load()
    ruff_texts = text_splitter.split_documents(docs)
    ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name='ruff')
    ruff = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                       retriever=ruff_db.as_retriever())

    # Create the Agent
    # tools = [
    #     Tool(
    #         name='State of Union QA System',
    #         func=state_of_union.run,
    #         description="useful for when you need to answer questions about the most recent state of the union address."
    #                     "Input should be a fully formed question.",
    #     ),
    #     Tool(
    #         name='Ruff QA System',
    #         func=ruff.run,
    #         description="useful for when you need to answer questions about ruff (a python linter). "
    #                     "Input should be a fully formed question.",
    #     ),
    # ]
    #
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    #
    # agent.run("What did biden say about ketanji brown jackson in the state of the union address?")
    #
    # agent.run("Why use ruff over flake8?")
    #
    # # Use the Agent solely as a router
    # tools = [
    #     Tool(
    #         name='State of Union QA System',
    #         func=state_of_union.run,
    #         description="useful for when you need to answer questions about the most recent state of the union address."
    #                     "Input should be a fully formed question.",
    #         return_direct=True,
    #     ),
    #     Tool(
    #         name='Ruff QA System',
    #         func=ruff.run,
    #         description="useful for when you need to answer questions about ruff (a python linter). "
    #                     "Input should be a fully formed question.",
    #         return_direct=True,
    #     ),
    # ]
    #
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    #
    # agent.run("What did biden say about ketanji brown jackson in the state of the union address?")
    #
    # agent.run("Why use ruff over flake8?")

    # Multi-Hop vectorstore reasoning
    tools = [
        Tool(
            name='State of Union QA System',
            func=state_of_union.run,
            description="useful for when you need to answer questions about the most recent state of the union address."
                        "Input should be a fully formed question.",
        ),
        Tool(
            name='Ruff QA System',
            func=ruff.run,
            description="useful for when you need to answer questions about ruff (a python linter). "
                        "Input should be a fully formed question.",
        ),
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    agent.run("What tool does ruff use to run over Jupyter Notebooks? Did the president "
              "mention that tool in the state of the union?")

