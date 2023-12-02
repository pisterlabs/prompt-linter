import logging
from typing import Dict, List, Optional

from langchain.agents import Tool
from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseChatMessageHistory
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from expert_gpts.database import get_db_session
from expert_gpts.database.expert_agents import ExpertAgentToolPrompt
from expert_gpts.embeddings.factory import EmbeddingsHandlerFactory
from expert_gpts.llms.chat_managers import SingleChatManager
from expert_gpts.llms.providers.openai import OpenAIApiManager
from shared.config import ExpertItem
from shared.llm_manager_base import BaseLLMManager
from shared.llms.openai import GPT_3_5_TURBO
from shared.llms.system_prompts import (
    PROMPT_ENGINEER_HUMAN_PROMPT,
    PROMPT_ENGINEER_SYSTEM_PROMPT,
    PROMPT_TOOL_ENGINEER_HUMAN_PROMPT,
    PROMPT_TOOL_ENGINEER_SYSTEM_PROMPT,
)
from shared.patterns import Singleton

AGENT_TOOL_GENERATOR_MODEL = GPT_3_5_TURBO
MAX_TOKENS = 1000
TEMPERATURE = 1
logger = logging.getLogger(__name__)


class ExpertAgentManager(metaclass=Singleton):
    tool_generator_prompt = ChatPromptTemplate.from_messages(
        [
            PROMPT_TOOL_ENGINEER_SYSTEM_PROMPT,
            PROMPT_TOOL_ENGINEER_HUMAN_PROMPT,
        ]
    )

    system_prompt_expert_prompt = ChatPromptTemplate.from_messages(
        [
            PROMPT_ENGINEER_SYSTEM_PROMPT,
            PROMPT_ENGINEER_HUMAN_PROMPT,
        ]
    )

    def __init__(self, api: Optional[BaseLLMManager] = OpenAIApiManager()):
        self.api = api

    def get_experts_as_agent_tools(
        self,
        experts_config: Dict[str, Optional[ExpertItem]],
        key_prefix: str = "",
        session_id: str = "same-session",
        memory: Optional[BaseMemory] = None,
        history: Optional[BaseChatMessageHistory] = None,
    ) -> List[Tool]:
        tools = []
        with get_db_session() as session:
            for expert_key, expert in experts_config.items():
                expert_key_with_prefix = f"{key_prefix}:{expert_key}"
                logger.debug(f"Creating agent tool for {expert_key_with_prefix}")
                stmt_expert = select(ExpertAgentToolPrompt).where(
                    ExpertAgentToolPrompt.expert_agent_key == expert_key_with_prefix
                )
                expert_model = ExpertAgentToolPrompt()
                expert_model.expert_agent_tool_prompt = expert.prompts.tool
                expert_model.expert_agent_key = expert_key_with_prefix
                if not expert.prompts.tool:
                    try:
                        expert_model = session.scalars(stmt_expert).one()
                    except NoResultFound:
                        expert_prompt_to_transform = expert.prompts.system
                        agent_tool_description = (
                            self.get_langchain_tools_expert_completion(
                                expert_prompt_to_transform
                            )
                        )
                        expert_model.expert_agent_tool_prompt = agent_tool_description
                        logger.warning(
                            f"Created agent tool for {expert_key_with_prefix} with description: "
                            f"{expert_model.expert_agent_tool_prompt}"
                        )
                session.add(expert_model)

                chat = self.get_expert_chat(
                    experts_config,
                    llm_manager=self.api,
                    embeddings_factory=EmbeddingsHandlerFactory(),
                    expert_key=expert_key,
                    session_id=session_id,
                    memory=memory,
                    history=history,
                )

                tools.append(
                    Tool(
                        name=expert_key,
                        func=lambda q: chat.ask(q),
                        description=expert_model.expert_agent_tool_prompt,
                        return_direct=expert.tool_return_direct,
                    )
                )

            session.commit()
        return tools

    def get_langchain_tools_expert_completion(self, expert_prompt_to_transform):
        agent_tool_description = self.api.create_chat_completion(
            self.tool_generator_prompt.format_messages(text=expert_prompt_to_transform),
            model=AGENT_TOOL_GENERATOR_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return agent_tool_description

    def optimize_prompt(self, prompt: str) -> str:
        logger.debug(f"Optimizing prompt: {prompt}")
        optimized_prompt = self.api.create_chat_completion(
            self.system_prompt_expert_prompt.format_messages(text=prompt),
            model=AGENT_TOOL_GENERATOR_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        logger.debug(f"Optimized prompt: {optimized_prompt}")
        return optimized_prompt

    def get_expert_chat(
        self,
        experts_map: Dict[str, ExpertItem],
        llm_manager: BaseLLMManager,
        embeddings_factory: EmbeddingsHandlerFactory,
        expert_key: str,
        session_id: str = "same-session",
        history: Optional[BaseChatMessageHistory] = None,
        memory: Optional[BaseChatMemory] = None,
    ):
        expert_config = [
            expert_config
            for dict_expert_key, expert_config in experts_map.items()
            if dict_expert_key == expert_key
        ][0]
        return SingleChatManager(
            llm_manager,
            expert_key,
            expert_config=expert_config,
            session_id=session_id,
            embeddings=embeddings_factory.get_expert_embeddings(
                llm_manager, expert_key, expert_config.embeddings.__root__
            ),
            query_embeddings_before_ask=expert_config.query_embeddings_before_ask,
            create_standalone_question_to_search_context=expert_config.create_standalone_question_to_search_context,
            history=history,
            memory=memory,
        )
