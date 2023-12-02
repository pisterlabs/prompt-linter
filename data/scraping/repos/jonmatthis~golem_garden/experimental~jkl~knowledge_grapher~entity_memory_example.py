# Langchain entity that has a conversation with you,
#   builds knowledge graph memories
#   outputs knowledge graphs in mermaid to a text file

# LEFTOFF: a little flummoxed at some of the ways agents get defined and run
# some parts seem incompatible in how they interact-- this simple example can't run a basic search tool
# this agent is also fake, so that needs some adjustment
# but also, it seemed like

from dotenv import load_dotenv
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv()

from pprint import pprint

from langchain.chat_models import ChatOpenAI

from langchain import ConversationChain

from langchain.agents import Tool
from langchain.tools import DuckDuckGoSearchRun

from langchain.memory import ConversationEntityMemory, ConversationBufferMemory

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world."
    ),
    # WriteFileTool(),
    # ReadFileTool(),
]

conversation_llm = ChatOpenAI(model="gpt-4", temperature=0.9)
memory_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

entity_memory = ConversationEntityMemory(llm=memory_llm)
buffer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationChain(
    llm=conversation_llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=entity_memory
)

pprint(chain.memory.entity_store.store)
