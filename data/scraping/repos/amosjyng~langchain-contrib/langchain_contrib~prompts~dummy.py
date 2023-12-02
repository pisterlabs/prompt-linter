"""Module defining a dummy prompt template."""

from typing import Any, List

from langchain.schema import PromptValue

from .z_base import ZBasePromptTemplate


class DummyPromptTemplate(ZBasePromptTemplate):
    """Dummy template for when you need a template but don't care for a real one."""

    input_variables: List[str] = []

    def format(self, **kwargs: Any) -> str:
        """Error out because this is a dummy prompt template."""
        raise NotImplementedError("You're using the dummy prompt template.")

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Error out because this is a dummy prompt template."""
        raise NotImplementedError("You're using the dummy prompt template.")

    @property
    def _prompt_type(self) -> str:
        """Dummy prompt type."""
        return "dummy"
