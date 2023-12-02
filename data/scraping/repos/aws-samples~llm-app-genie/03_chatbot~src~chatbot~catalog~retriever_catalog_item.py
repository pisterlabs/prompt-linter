""" Module that contains an abstract base class that represents a retriever catalog item.
"""
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from chatbot.llm_app import BaseLLMApp, LLMApp, RAGApp
from langchain.schema import BaseRetriever

from .catalog import CatalogById
from .catalog_item import CatalogItem
from .model_catalog_item import ModelCatalogItem


@dataclass
class RetrieverCatalogItem(CatalogItem[BaseRetriever]):
    """Abstract base class that represents a retriever catalog item."""

    @property
    def available_filter_options(self) -> Union[List[Tuple[str, Any]], None]:
        return None

    @property
    def current_filter(self) -> List[Tuple[str, Any]]:
        return []

    @current_filter.setter
    def current_filter(self, value: List[Tuple[str, Any]]):
        pass
