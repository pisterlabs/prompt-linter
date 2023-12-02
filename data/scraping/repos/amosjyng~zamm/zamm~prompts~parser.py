from abc import abstractmethod
from typing import Dict

from langchain.prompts.base import BaseOutputParser


class DictOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a dict."""

    @abstractmethod
    def parse(self, text: str) -> Dict[str, str]:
        """Parse the output of an LLM call."""
