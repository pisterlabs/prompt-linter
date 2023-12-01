from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss

from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
import chainlit as cl

# The search tool has no async implementation, we fall back to sync
@cl.langchain_factory(use_async=False)
def agent():
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)

    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    callbacks = [cl.LangchainCallbackHandler()]
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Xiaoming",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(
            temperature=0, 
            streaming=True, 
            callbacks=callbacks
        ),
        memory=vectorstore.as_retriever(),
    )
    # Set verbose to be true
    agent.chain.verbose = True
    agent.chain.callbacks = callbacks
    return agent


@cl.langchain_run
async def run(agent, input):
    # Since the agent is sync, we need to make it async
    res = await cl.make_async(agent.run)([input])
    await cl.Message(content=res).send()
