"""For PoC purposes, I am largely just copying the LangChain AutoGPT example found here:
    https://python.langchain.com/en/latest/use_cases/autonomous_agents/autogpt.html

    Essentially, I am leveraging LangChain's experiemental AutoGPT agent to create a simple
    autonomous agent that can help me manage my schedule and task list better.  I am adding
    in Google Calendar tools, since that is the only LangChain doesn't have out of the box.
    
    Additionally, this agent will be integrated into a flask app that will, initially, be
    used as the events endpoints for a slack app"""

from langchain.agents import Tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.utilities import SerpAPIWrapper

from bespokebots.services.agent.google_calendar_tools import (
    GoogleCalendarCreateEventTool, 
    GoogleCalendarViewEventsTool,
    GoogleCalendarUpdateEventTool,
    GoogleCalendarDeleteEventTool
    )
from bespokebots.services.chains import CalendarDataAnalyzerTool

#Set up the tools for the agent
#search = SerpAPIWrapper()
tools = [
    # Tool(
    #     name = "search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events. You should ask targeted questions"
    # ),
    WriteFileTool(),
    ReadFileTool(),
    GoogleCalendarCreateEventTool(),
    GoogleCalendarViewEventsTool(),
    GoogleCalendarUpdateEventTool(),
    GoogleCalendarDeleteEventTool(),
    CalendarDataAnalyzerTool()
]

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
#Set up the agent's memory
from langchain.vectorstores import FAISS

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
import faiss

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

from langchain.chat_models import ChatOpenAI
#Set up the agent
from langchain.experimental import AutoGPT

def build_agent():
    agent = AutoGPT.from_llm_and_tools(
        ai_name="Tom",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"),
        memory=vectorstore.as_retriever(),
        human_in_the_loop = False
    )
    # Set verbose to be true
    agent.chain.verbose = True
    return agent