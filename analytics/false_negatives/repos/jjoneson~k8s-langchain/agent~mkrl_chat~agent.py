from typing import List, Optional, Sequence
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools.base import BaseTool
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX


class ZeroShotChatAgent(ZeroShotAgent):
    @classmethod
    def create_chat_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> ChatPromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(tool_names=tool_names)
        system_template = "\n\n".join([prefix, tool_strings, format_instructions])
        system_message_template = SystemMessagePromptTemplate.from_template(system_template)
        human_message_template = HumanMessagePromptTemplate.from_template(suffix)
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
            return ChatPromptTemplate.from_messages(
                [system_message_template, human_message_template],
            )