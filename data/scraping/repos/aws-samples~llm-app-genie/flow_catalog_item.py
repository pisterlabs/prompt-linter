""" Module that contains an abstract base class that represents a retriever catalog item.
"""
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from chatbot.llm_app import BaseLLMApp, LLMApp, RAGApp
from langchain.chains.base import Chain
from abc import ABC
from abc import abstractmethod

from .retriever_catalog_item import RetrieverCatalogItem
from .catalog import CatalogById
from .catalog_item import CatalogItem
from .model_catalog_item import ModelCatalogItem
from .agent_chain_catalog_item import AgentChainCatalogItem

@dataclass
class FlowCatalogItem(CatalogItem[Chain], ABC):
    """Abstract base class that represents a retriever catalog item."""

    @property
    def available_filter_options(self) -> Union[List[Tuple[str, Any]], None]:
        return None

    @property
    def enable_retriever(self) -> bool:
        return False

    @property
    def enable_file_upload(self) -> bool:
        return False

    @property
    def enable_agents_chains(self) -> bool:
        return False

    @property
    def current_filter(self) -> List[Tuple[str, Any]]:
        return []

    @current_filter.setter
    def current_filter(self, value: List[Tuple[str, Any]]):
        pass

    @abstractmethod
    def llm_app_factory(
        self, 
        retriever: RetrieverCatalogItem,
        model: ModelCatalogItem, 
        agent_chain: AgentChainCatalogItem,
        prompt_catalog: CatalogById,
        sql_connection_uri: str,
        sql_model: ModelCatalogItem
    ) -> LLMApp:
        """
        Returns the llm app to use for this retriever.
        """
        rag_prompt = prompt_catalog[model.rag_prompt_identifier].get_instance()
        llm = model.get_instance()
        condense_question_prompt = prompt_catalog[
            "prompts/condense_question.yaml"
        ].get_instance()

        retriever = retriever.get_instance()

        return RAGApp(
            prompt=rag_prompt,
            llm=llm,
            condense_question_prompt_template=condense_question_prompt,
            retriever=retriever,
        )
