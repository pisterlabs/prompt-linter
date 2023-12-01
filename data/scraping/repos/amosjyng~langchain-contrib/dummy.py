"""Module that defines a dummy agent."""

from typing import Dict, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.tools import BaseTool
from pydantic import root_validator

from langchain_contrib.chains import DummyLLMChain


class DummyAgent(Agent):
    """Used when you want to use langchain's AgentExecutor but not Agent."""

    llm_chain: LLMChain = DummyLLMChain()

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract tool and tool input from llm output."""
        raise NotImplementedError("You're using the dummy Agent")

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        raise NotImplementedError("You're using the dummy Agent")

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        raise NotImplementedError("You're using the dummy Agent")

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Create a prompt for this class."""
        raise NotImplementedError("You're using the dummy Agent")

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """Ignore parent prompt validation."""
        raise NotImplementedError("You're using the dummy Agent")

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "dummy"
