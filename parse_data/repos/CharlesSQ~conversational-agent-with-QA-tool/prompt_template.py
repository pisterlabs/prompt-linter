from typing import List
from langchain.agents import Tool
from langchain.schema import SystemMessage
from langchain.prompts import BaseChatPromptTemplate


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    prefix: str
    instructions: str
    sufix: str
    # The list of tools available
    tools: List[Tool]

    def _set_tool_description(self, tool_description, tool_name, tool_input):
        full_description = f"""{tool_description}, send this:
```json
{{"action": "{tool_name}",
"action_input": "{tool_input}"}}
```
"""
        return full_description

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""

        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        separator = '. Input:'
        kwargs["tools"] = "\n".join(
            [f"{self._set_tool_description(tool.description.split(separator)[0], tool.name, tool.description.split(separator)[1])}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        # Format the instructions replacing the variables with the values
        formatted = self.instructions.format(**kwargs)

        # Add the sufix
        formatted += self.sufix
        return [SystemMessage(content=formatted)]
