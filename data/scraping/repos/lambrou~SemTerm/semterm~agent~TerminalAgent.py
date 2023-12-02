import os
from abc import ABC
from typing import Sequence, Optional, List, Tuple, Any
from langchain import BasePromptTemplate
from langchain.agents import (
    ConversationalChatAgent,
    AgentOutputParser,
)
from langchain.tools import BaseTool
from pydantic import Field

from .TerminalAgentPrompt import (
    PREFIX,
    SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
)
from langchain.schema import (
    AgentAction,
    BaseOutputParser,
    BaseMessage,
    AIMessage,
    SystemMessage,
)
from semterm.terminal.TerminalOutputParser import TerminalOutputParser


class TerminalAgent(ConversationalChatAgent, ABC):
    output_parser: AgentOutputParser = Field(default_factory=TerminalOutputParser)

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX.format(current_directory=os.getcwd()),
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        return super().create_prompt(
            tools=tools,
            system_message=system_message,
            human_message=human_message,
            input_variables=input_variables,
            output_parser=output_parser or cls._get_default_output_parser(),
        )

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            if action.tool == "Human":
                thoughts.append(AIMessage(content=action.tool_input))
                continue
            if isinstance(action.tool_input, list):
                observation = observation.replace(";".join(action.tool_input), "")
            else:
                observation = observation.replace(action.tool_input, "")
            thoughts.append(AIMessage(content=action.log))
            system_message = SystemMessage(
                content=TEMPLATE_TOOL_RESPONSE.format(observation=observation)
            )
            thoughts.append(system_message)
        return thoughts
