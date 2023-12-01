""" Abstract base class that represents a prompt item.
"""
from dataclasses import dataclass

from langchain.prompts import BasePromptTemplate

from .catalog_item import CatalogItem


@dataclass
class PromptCatalogItem(CatalogItem[BasePromptTemplate]):
    """Represents a prompt template."""

    prompt: BasePromptTemplate
    """Prompt template"""

    def get_instance(self) -> BasePromptTemplate:
        return self.prompt
