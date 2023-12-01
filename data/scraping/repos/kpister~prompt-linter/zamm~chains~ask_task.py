"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from typing import Dict, List

from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel

from .bash_action_prompt import MANAGER_TEMPLATE


class AskForTaskChain(Chain, BaseModel):
    """Ask for a task to perform"""

    llm: BaseLLM
    """LLM wrapper to use."""
    prompt: BasePromptTemplate = MANAGER_TEMPLATE
    output_key: str = "task"

    @property
    def input_keys(self) -> List[str]:
        """Input keys to this chain."""
        return []

    @property
    def output_keys(self) -> List[str]:
        """Output keys produced by this chain."""
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output = self.llm(self.prompt.format(**inputs), stop=["\n"]).strip()

        return {self.output_key: output}
