"""Attempt to implement MRKL systems as described in arxiv.org/pdf/2205.00445.pdf."""
from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Optional, Tuple

from langchain.agents.agent import Agent
from langchain.input import print_text
from backend.modules.agents.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX, HISTORY, RESOURCES
from backend.modules.resources.resource import Resource
from langchain.agents.tools import Tool
from langchain.llms.base import LLM
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from backend.modules.resources.resource import format_resources

FINAL_ANSWER_ACTION = "Final Answer: "


class ChainConfig(NamedTuple):
    """Configuration for chain to use in MRKL system.

    Args:
        action_name: Name of the action.
        action: Action function to call.
        action_description: Description of the action.
    """

    action_name: str
    action: Callable
    action_description: str


def get_action_and_input(llm_output: str) -> Tuple[str, str]:
    """Parse out the action and input from the LLM output."""
    ps = [p for p in llm_output.split("\n") if p]
    if ps[-1].startswith("Final Answer"):
        directive = ps[-1][len(FINAL_ANSWER_ACTION):]
        return "Final Answer", directive
    if not ps[-1].startswith("Action Input: "):
        raise ValueError(
            "The last line does not have an action input, "
            "something has gone terribly wrong."
        )
    if not ps[-2].startswith("Action: "):
        raise ValueError(
            "The second to last line does not have an action, "
            "something has gone terribly wrong."
        )
    action = ps[-2][len("Action: "):]
    action_input = ps[-1][len("Action Input: "):]
    return action, action_input.strip(" ").strip('"')


class ZeroShotAgent(Agent):
    """Agent for the MRKL chain."""

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: List[Tool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
        template = "\n\n".join(
            [prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @classmethod
    def _validate_tools(cls, tools: List[Tool]) -> None:
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )

    def _extract_tool_and_input(self, text: str) -> Optional[Tuple[str, str]]:
        return get_action_and_input(text)


class ReActMemoryAgent(ZeroShotAgent):

    @classmethod
    def create_prompt(
        cls,
        tools: List[Tool],
        resources: List[Resource],
        history: str,
        debug: bool = False,
    ) -> PromptTemplate:
        tool_strings = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])

        def process(text: str):
            return text.replace("{", "{{").replace("}", "}}")

        resources_strings = format_resources(resources)
        resources_string = RESOURCES.format(
            resources=process(resources_strings))

        history = HISTORY.format(history=process(history))

        format_instructions = FORMAT_INSTRUCTIONS.format(
            tool_names=process(tool_names))
        template = "\n\n".join(
            [PREFIX, tool_strings, format_instructions, history, resources_string, SUFFIX])
        if debug:
            print_text(f"{cls.__name__} prompt:\n")
            print_text(template, color="blue")
        input_variables = ["input"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @ classmethod
    def from_llm_tools_resources_history(
        cls,
        llm: LLM,
        tools: List[Tool],
        resources: List[Resource],
        history: str,
        debug: bool = False,
        **kwargs: Any
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        llm_chain = LLMChain(
            llm=llm, prompt=cls.create_prompt(tools, resources, history, debug))
        return ReActMemoryAgent(llm_chain=llm_chain, tools=tools, **kwargs)
