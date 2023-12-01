from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from typing import List
from langchain.schema import HumanMessage

PREFIX = "Answer the following question as best you can. You have access to the following tools:"
SUFFIX = """Begin!
Question: {input}
Thought:{agent_scratchpad}
Response as valid JSON object {open_brace} "thought": "<mandatory>", "action": "<mandatory>", "actionInput": "<mandatory>" {close_brace}:
"""

instructions = """The response is a valid JSON object in the following format:

{open_brace}
    "thought": "(Provide your thought process here)",
    "action": "(Specify the action you will take, i.e. choose one of {tool_names})",
    "actionInput": "(Provide the input for the action in the required format)"
{close_brace}
    
In the next iteration you will be given the Observation from the previous iteration.
You can then choose the tool to give the final answer, or you can continue to use the other tools to gather more observations.
"""

class JsonPromptTemplate(BaseChatPromptTemplate):
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nStart Tool Observation: {observation}\nEnd Tool Observation\nThought: "
        kwargs["agent_scratchpad"] = thoughts

        tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_strings += "\nFinalResponse: The final response to the original input question"

        tool_names = ", ".join([tool.name for tool in self.tools])
        tool_names += ", FinalResponse"

        template = "\n\n".join([PREFIX, tool_strings, instructions, SUFFIX])

        kwargs["tool_names"] = tool_names
        kwargs["open_brace"] = "{"
        kwargs["close_brace"] = "}"

        formatted = template.format(**kwargs)
        return [HumanMessage(content=formatted)]
