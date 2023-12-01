from typing import Dict, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate
from langchain.tools import BaseTool
from pydantic import root_validator

from zamm.chains.dummy import DummyLLMChain


class CustomAgent(Agent):
    """Custom agent that sidesteps a lot of langchain's opinions about how agents
    should work."""

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract tool and tool input from llm output."""
        raise NotImplementedError()

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        raise NotImplementedError()

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        raise NotImplementedError()

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Create a prompt for this class."""
        raise NotImplementedError()

    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """Ignore parent prompt validation"""
        return values

    @classmethod
    def from_llm(cls, llm: BaseLLM):
        return cls(llm_chain=DummyLLMChain(llm=llm))


class DummyAgent(CustomAgent):
    """Used when you want to use langchain's AgentExecutor but not Agent"""

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "zamm-dummy"
