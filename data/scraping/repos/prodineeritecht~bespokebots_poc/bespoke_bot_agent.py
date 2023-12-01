from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError
import os
import langchain
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.tools.base import BaseTool
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks import StdOutCallbackHandler
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import initialize_agent
from langchain.prompts import MessagesPlaceholder
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor,StructuredChatAgent
from langchain.agents import AgentType
from langchain.callbacks import get_openai_callback
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss

from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.utilities import SerpAPIWrapper

from bespokebots.dao.database import db

from bespokebots.services.agent.google_calendar_tools import (
    GoogleCalendarCreateEventTool,
    GoogleCalendarViewEventsTool,
    GoogleCalendarUpdateEventTool,
    GoogleCalendarDeleteEventTool,
)
from bespokebots.services.chains.output_parsers import (
    CalendarAnalyzerOutputParserFactory
)
from bespokebots.services.chains import CalendarDataAnalyzerTool
from bespokebots.models import (
    CommunicationChannel
)

from bespokebots.services.agent.todoist_tools import (
    CreateTaskTool,
    CloseTaskTool,
    ViewProjectsTool,
    CreateProjectTool,
    GetProjectIdsTool,
    FindTasksWithFilterTool
) 


# Things the ai assistant needs to be able to do
# 1. Take in a user goal
# 2. Take in a list of tools
# 3. Take in an LLM Model, will default to OpenAI GPT-4
# 4. Take in a vector store, will default to FAISS maybe?
# 5. Take in a prompt
# 6. Take in an outputparser, will default to a custom output parser
# 7. Execute the agent's LLMChain in a loop until the user goal is met
#   a. The loop will also be built with either a timeout or a max number of iterations

class Singleton:
    _instances = {}

    @classmethod
    def instance(cls):
        if cls not in cls._instances:
            cls._instances[cls] = cls()
        return cls._instances[cls]



class BespokeBotAgent(Singleton):
    """The BespokeBotAgent class is responsible for handling the agent mechanics required for the digital assistant.
    This class is really a wrapper around the StructuredChatAgent class from langchain. It is responsible for
    setting up the tools, the LLMChain, and the agent itself. It also handles the execution of the agent.

    Need to figure out how to create an instance of the bespoke bot agent for each user, and then
    keep that instance persisted for each user. We don't want one user's interactions to bleed into another user's interactions.
    In addition to user privacy concerns, bleeding interactions would also be a bad user experience.
    This will need to be figured out as part of the user architecture when I add in the persistence layer.
    """
    # Default vector store
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    

    # turn on tracing for the agent
    os.environ["LANGCHAIN_TRACING"] = "false"
    langchain.debug = True
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    

    def __init__(
        self,
        ai_name: str = "BespokeBot",
        llm_model: str = "gpt-4",
        temperature: float = 0.0,
        memory: VectorStoreRetriever = None,
        additional_tools: List[BaseTool] = None,
        callback_manager: Optional[BaseCallbackManager] = None,
        feedback_tool: Optional[HumanInputRun] = None,
    ):
        self.ai_name = ai_name
        self.memory = BespokeBotAgent.vectorstore if memory is None else memory
        self.full_message_history: List[BaseMessage] = []
        self.next_action_count = 0
        self.additional_tools = additional_tools
        self.feedback_tool = feedback_tool
        self.agent = None
        self.prompt = None
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm_chain = None
        self.llm = None
        self.executor = None
        self.prefix = None
        self.suffix = None
        self.initialized = False

    def _setup_tools(self):
        # Set up the tools for the agent
        #search = SerpAPIWrapper()
        tools = [
            # Tool(
            #     name="search",
            #     func=search.run,
            #     description="useful for when you need to answer questions about current events. You should ask targeted questions",
            # ),
            GoogleCalendarCreateEventTool(),
            GoogleCalendarViewEventsTool(),
            GoogleCalendarUpdateEventTool(),
            GoogleCalendarDeleteEventTool(),
            CalendarDataAnalyzerTool(return_direct = True),
            ViewProjectsTool(),
            CreateProjectTool(),
            CreateTaskTool(),
            CloseTaskTool(),
            GetProjectIdsTool(),
            FindTasksWithFilterTool(),
        ]
        return tools
    
    def is_initialized(self):
        return self.initialized

    @staticmethod
    def get_agent(prefix, suffix, input_variables: List[str] = None):
        agent = BespokeBotAgent.instance()
        if not agent.is_initialized():
            agent.initialize_agent(prefix=prefix, suffix=suffix, input_variables=input_variables)
            agent.initialized = True
        return agent

    # def estimate_tokens(self, message: str) -> int:
    #     """Estimate the number of tokens required to generate a response."""
    #     token_usage = ""
    #     with get_openai_callback() as callback:
    #         response self.llm()
        
    #     return self.agent.estimate_tokens(message)
    
    def initialize_agent(
        self, prefix: str, suffix: str, input_variables: List[str] = None
    ) -> AgentExecutor:
        """Initialize the agent with the tools and prompt."""
        self.tools = self._setup_tools()
        if self.additional_tools:
            self.tools.extend(self.additional_tools)

        self.llm = ChatOpenAI(temperature=self.temperature, model_name=self.llm_model)

        convo_memory = ConversationBufferMemory(memory_key="history",return_messages=True)
        chat_history = MessagesPlaceholder(variable_name="history") 
        self.executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=convo_memory,
            verbose=True,
            agent_kwargs= {
                'prefix' : prefix,
                'suffix' : suffix,
                'memory_prompts': [chat_history],
                'input_variables': ["input", "agent_scratchpad", "history"]
            }
        )

    def run_agent(self, user_goal: str, user_id: str) -> str:
        """Run the agent with a user goal."""
        
        if self.executor is None:
            self.initialize_agent(prefix=self.prefix, suffix=self.suffix)
        
        user_goal_with_id = f"My user id is {user_id}, use this value for any tools that require a 'user_id' parameter. \\n {user_goal}"
        
        return self.executor.run(user_goal_with_id)
