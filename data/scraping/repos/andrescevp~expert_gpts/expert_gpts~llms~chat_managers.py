import logging
from functools import lru_cache
from typing import List, Literal, Optional

from langchain.agents import Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage
from langchain.schema import BaseChatMessageHistory

from expert_gpts.chat_history.mysql import MysqlChatMessageHistory
from expert_gpts.database import get_db_session
from expert_gpts.embeddings.base import EmbeddingsHandlerBase
from shared.config import ExpertItem
from shared.llm_manager_base import BaseLLMManager
from shared.llms.system_prompts import (
    CHAT_HUMAN_PROMPT_TEMPLATE,
    CHAT_SYSTEM_PROMPT_STANDALONE_QUESTION,
)

DEFAULT_EXPERT_CONFIG = ExpertItem()

logger = logging.getLogger(__name__)


@lru_cache
def get_history(session_id, ai_key):
    with get_db_session() as session:
        return MysqlChatMessageHistory(
            session_id=session_id, session=session, ai_key=ai_key
        )


TYPE_MEMORY_TYPE = Literal["default", "summary"]


@lru_cache
def get_memory(
    llm: Optional[BaseLLMManager],
    memory_type: TYPE_MEMORY_TYPE = "default",
    memory_key: str = "chat_history",
    chat_memory: Optional[BaseChatMessageHistory] = None,
):
    args = {
        "llm": llm,
        "memory_key": memory_key,
        "return_messages": True,
    }
    if chat_memory is not None:
        args["chat_memory"] = chat_memory

    if memory_type == "summary":
        return ConversationSummaryBufferMemory(**args)

    return ConversationBufferMemory(**args)


class SingleChatManager:
    _instances = {}

    def __init__(
        self,
        llm_manager: BaseLLMManager,
        expert_key: str,
        expert_config: ExpertItem = DEFAULT_EXPERT_CONFIG,
        session_id: str = "same-session",
        memory_key: str = "chat_history",
        memory_type: TYPE_MEMORY_TYPE = "default",
        embeddings: Optional[EmbeddingsHandlerBase] = None,
        create_standalone_question_to_search_context: bool = True,
        query_embeddings_before_ask: bool = True,
        history: Optional[BaseChatMessageHistory] = None,
        memory: Optional[BaseChatMemory] = None,
    ):
        self.create_standalone_question_to_search_context = (
            create_standalone_question_to_search_context
        )
        self.query_embeddings_before_ask = query_embeddings_before_ask
        self.embeddings = embeddings
        self.expert_config = expert_config
        self.expert_key = expert_key
        self.llm_manager = llm_manager
        self.session_id = session_id
        self.history = history if history else get_history(session_id, expert_key)
        self.memory = (
            memory
            if memory
            else get_memory(
                llm_manager,
                chat_memory=self.history,
                memory_type=memory_type,
                memory_key=memory_key,
            )
        )

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        cls_key = None
        if cls_key not in cls._instances:
            cls_key = f"expert_key_{kwargs['expert_key']}_{kwargs['session_id']}"

            cls._instances[cls_key] = cls(*args, **kwargs)

        return cls._instances[cls_key]

    def ask(self, question):
        search_context_question = question
        if self.create_standalone_question_to_search_context:
            search_context_question = get_standalone_question(
                question,
                self.history.messages[:5],
                self.llm_manager,
                self.expert_config.temperature,
                self.expert_config.max_tokens,
                self.expert_config.model,
            )

        context = ""
        if self.embeddings and self.query_embeddings_before_ask:
            try:
                context = self.embeddings.search(search_context_question).response
            except Exception as e:
                logger.error("Could not query embeddings: %s", e)

        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.expert_config.prompts.system,
                    name=f"{self.expert_key}ChatBot",
                    role="system",
                ),
                CHAT_HUMAN_PROMPT_TEMPLATE,
            ]
        )

        answer = self.llm_manager.create_chat_completion(
            template.format_messages(
                question=question,
                context=context,
            ),
            temperature=self.expert_config.temperature,
            max_tokens=self.expert_config.max_tokens,
            model=self.expert_config.model,
        )

        return answer

    def get_log(self):
        return self.llm_manager.callbacks_handler.log


class ChainChatManager:
    _instances = {}

    def __init__(
        self,
        llm_manager: BaseLLMManager,
        temperature: float = 0,
        max_tokens: int | None = None,
        chain_key: str = "default",
        tools: Optional[List[Tool]] = None,
        model: str | None = None,
        session_id: str = "same-session",
        memory_key: str = "chat_history",
        memory_type: TYPE_MEMORY_TYPE = "default",
        agent_type: AgentType = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        create_standalone_question_to_search_context: bool = True,
        history: Optional[BaseChatMessageHistory] = None,
        memory: Optional[BaseChatMemory] = None,
    ):
        self.chain_key = chain_key
        self.create_standalone_question_to_search_context = (
            create_standalone_question_to_search_context
        )
        self.agent_type = agent_type
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_manager = llm_manager
        self.session_id = session_id
        self.history = history if history else get_history(session_id, chain_key)
        self.memory = (
            memory
            if memory
            else get_memory(
                llm_manager,
                chat_memory=self.history,
                memory_key=memory_key,
                memory_type=memory_type,
            )
        )

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        cls_key = None
        if cls_key not in cls._instances:
            cls_key = f"expert_key_{kwargs['chain_key']}_{kwargs['session_id']}"

            cls._instances[cls_key] = cls(*args, **kwargs)

        return cls._instances[cls_key]

    def ask(self, question):
        answer = self.llm_manager.create_chat_completion_with_agent(
            question,
            agent_type=self.agent_type,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            tools=self.tools,
            memory=self.memory,
            agent_key=self.chain_key,
        )

        return answer

    def get_log(self):
        return self.llm_manager.callbacks_handler.log


class PlannerManager:
    _instances = {}

    def __init__(
        self,
        llm_manager: BaseLLMManager,
        temperature: float = 0,
        max_tokens: int | None = None,
        chain_key: str = "default",
        tools: Optional[List[Tool]] = None,
        model: str | None = None,
    ):
        self.chain_key = chain_key
        self.model = model
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.llm_manager = llm_manager

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        cls_key = None
        if cls_key not in cls._instances:
            cls_key = f"planner_key_{kwargs['chain_key']}_{kwargs['session_id']}"

            cls._instances[cls_key] = cls(*args, **kwargs)

        return cls._instances[cls_key]

    def ask(self, question):
        answer = self.llm_manager.execute_plan(
            question,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.tools,
            agent_key=self.chain_key,
        )

        return answer

    def get_log(self):
        return self.llm_manager.callbacks_handler.log


def get_standalone_question(
    question, chat_history, llm_manager, temperature, max_tokens, model
):
    template = ChatPromptTemplate.from_messages(
        [
            CHAT_SYSTEM_PROMPT_STANDALONE_QUESTION,
            CHAT_HUMAN_PROMPT_TEMPLATE,
        ]
    )
    return llm_manager.create_chat_completion(
        template.format_messages(
            question=question,
            context=chat_history,
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )
