"""Defines the Prefixed prompt template type."""
from __future__ import annotations

from typing import Optional

from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel, Extra

from .chained import ChainedPromptTemplate
from .schema import Templatable, into_template


class PrefixedTemplate(BaseModel):
    """Wraps another prompt template into one that can take in a prefix.

    This is useful for when you want to add a prefix to a prompt,
    but you don't want to have to do it manually.
    """

    template: BasePromptTemplate

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def __init__(self, templatable: Templatable) -> None:
        """Initialize the PrefixedTemplate.

        Args:
            templatable: The template to wrap.
        """
        super().__init__(template=into_template(templatable))

    def __call__(
        self, prefix: Optional[Templatable] = None, joiner: str = ""
    ) -> BasePromptTemplate:
        """Create the final PromptTemplate with a prefix.

        Args:
            prefix: The prefix to add to the existing template.
            joiner: The string to put in between the prefix and the template.

        Returns:
            The final template with the prefix added if specified.
        """
        if prefix:
            return ChainedPromptTemplate([prefix, self.template], joiner=joiner)
        return self.template
