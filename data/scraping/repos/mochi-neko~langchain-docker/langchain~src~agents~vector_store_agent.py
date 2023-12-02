import langchain.llms
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

def setup_agent(llm : langchain.llms.BaseLLM, memory: BaseMemory):
    # Setup vector store
    loader = TextLoader("state_of_the_union.txt", encoding="utf8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings, collection_name="state-of-the-union")

    state_of_union = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    # Setup tools
    tools = [
        Tool(
            name = "State of Union QA System",
            func=state_of_union.run,
            description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question."
        )
    ]

    # Setup agent
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True)

    return agent_executor