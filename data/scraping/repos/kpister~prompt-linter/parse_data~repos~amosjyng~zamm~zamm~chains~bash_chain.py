"""Chain that interprets a prompt and executes bash code to perform bash operations."""

from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel

from zamm.actions.use_terminal.chain import TerminalChain

from .bash_action_prompt import BASH_COMMAND_TEMPLATE


class ZBashChain(TerminalChain, BaseModel):
    """Custom Bash chain for easier experimentation"""

    llm: BaseLLM
    """LLM wrapper to use."""
    prompt: BasePromptTemplate = BASH_COMMAND_TEMPLATE
