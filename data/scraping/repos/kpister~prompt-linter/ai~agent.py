from typing import Awaitable, Callable, Coroutine, List, Optional
from pydantic import Extra

from langchain.callbacks import StdOutCallbackHandler, OpenAICallbackHandler, HumanApprovalCallbackHandler
from langchain.prompts import MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, BaseChatMessageHistory
from langchain.tools.base import ToolException, BaseTool

from ai.arxiv import LoadedPapersStore, PaperMetadata
from ai.prompts import AGENT_PROMPT, PAPERS_PROMPT
from ai.store import PaperStore
from ai.tools import ArxivSearchTool, AbstractSummaryTool, AbstractQuestionsTool, DiscordToolCallback, PaperBackend, PaperQATool, SummarizePaperTool, PaperCitationsTool, get_tools
from config import CONFIG




        
class ArxivAgent:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo-0613", callbacks=[OpenAICallbackHandler()])
    chat_llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0613", callbacks=[OpenAICallbackHandler()])
    
    def __init__(self, chat_window=CONFIG.CHAT_WINDOW, verbose=False):
        self.verbose = verbose

        # use discord message id as collection name
        # will persist documents loaded in a reply chain
        self.vectorstore = Chroma(
            collection_name="main", 
            embedding_function=self.embeddings,
            persist_directory="vectorstore"
        )
        num_vects = len(self.vectorstore.get()["documents"])
        print(f"Loaded collection `{self.vectorstore._collection.name}` from directory `{self.vectorstore._persist_directory}` with {num_vects} vector(s)")

        self.paper_store = PaperStore(CONFIG.PAPER_STORE_PATH)        
        self.chat_window = chat_window # max no. interactions (Human + AI messages) in chat history


        self.agent = self._init_agent()
    
    

    async def acall(self, input: str, chat_id: str, chat_history: BaseChatMessageHistory, handle_tool_msg: DiscordToolCallback.Callback):
        """Call the model with a new user message and its message history.
        Loaded papers will be automatically fetched.

        Args:
            input (str): next user message
            chat_id (str): client provided unique identifier of conversation
            chat_history (BaseChatMessageHistory): message history
        """
        memory = ConversationBufferWindowMemory(
            chat_memory=chat_history,
            return_messages=True,
            input_key="input",
            memory_key="memory",
            k=self.chat_window
        )

        # get tools with a backend
        # chat_id lets us update the mentioned papers list for a chat
        backend = PaperBackend(chat_id, self.vectorstore, self.paper_store, self.chat_llm)
        tools = get_tools(self._parse_tool_error, backend)
        
        exec_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            memory=memory,
            verbose=self.verbose
        )

        # papers mentioned in conversation
        papers = self.paper_store.get_papers(chat_id)

        return await exec_chain.arun(
            input=input,
            papers=papers,
            callbacks=[DiscordToolCallback(handle_tool_msg, tools)]
        )

    def _init_agent(self):
        prompt_template = PromptTemplate(
            input_variables=["papers"],
            template=PAPERS_PROMPT
        )
        papers_prompt = SystemMessagePromptTemplate(
            prompt=prompt_template
        )
        message_history = MessagesPlaceholder(variable_name="memory")
        extra_prompt_messages = [papers_prompt, message_history]
        system_message = SystemMessage(
            content=AGENT_PROMPT
        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            extra_prompt_messages=extra_prompt_messages,
            system_message=system_message
        )
        return OpenAIFunctionsAgent(
            llm=self.chat_llm,
            tools=get_tools(),
            prompt=prompt,
            verbose=self.verbose
        )
        
    def _get_loaded_papers_msg(self, paper_metas: List[PaperMetadata]):
        """Format metadata list for system prompt"""
        if len(paper_metas) > 0:
            return "\n".join([metadata.short_repr() for metadata in paper_metas])
        else:
            return "NONE"

    def _parse_tool_error(self, err: ToolException):
        return f"An error occurred: {err.args[0]}"
    
    def save(self):
        """Must call before exiting to save vectorstore and paper store"""
        self.paper_store.save()
    


