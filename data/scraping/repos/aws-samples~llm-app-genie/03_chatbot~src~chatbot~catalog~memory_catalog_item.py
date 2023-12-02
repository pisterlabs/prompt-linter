""" Class that represents an item in a memory catalog. """

from abc import abstractmethod
from dataclasses import dataclass

from langchain.schema import BaseChatMessageHistory

from .catalog_item import CatalogItem


@dataclass
class MemoryCatalogItem(CatalogItem[BaseChatMessageHistory]):
    """Abstract base class that represents a catalog item."""

    @abstractmethod
    def get_instance(self, session_id) -> BaseChatMessageHistory:
        """Returns an instance of the item."""
