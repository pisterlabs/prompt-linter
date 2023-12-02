from typing import List

from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import HumanMessage

from twk_backend.custom_chat_agent.utils import convert_entities_to_string

custom_template = """
Your name is {name}. {description}. {writingStyle}.
Have the following conversation with Human, respond  to them, and answer their questions as best you can.
You have access to the following tools:

{tools}

When responsing, use the following formats:

**Option 1:**
Use this if you want to use a tool:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**Option 2:**
Use this if you do not have to use a tool to properly respond to the user:
Final Answer: string \\ You should put what you want to return to use here

Begin! Remember while giving your final answer, your name is {name}.
{writingStyle}.

Context:
{entities}

Previous conversation history:
{history}

New Input:
Human: {input}
{agent_scratchpad}
"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str = custom_template
    # The list of tools available
    tools: List[Tool]
    # The writingStyle
    writingStyle: str
    # The name
    name: str
    # The description
    prompt: str

    def format_messages(self, **kwargs) -> str:
        # Set up basic values for templates
        kwargs["name"] = self.name
        kwargs["writingStyle"] = self.writingStyle
        kwargs["description"] = self.prompt
        kwargs["entities"] = convert_entities_to_string(kwargs["entities"])
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
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
