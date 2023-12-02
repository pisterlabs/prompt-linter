from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from typing import List
from langchain.schema import HumanMessage

# Set up the base template
template = """Answer the following question as best you can. You have access to the following tools:

{tools}

The response is given in the following format:

___start_response___
___start_thought___(Provide your thought process here)___end_thought___
___start_action___(Specify the action you will take, i.e. choose one of the {tool_count} available tools: {tool_names})___end_action___
___start_action_input___(Provide the input for the action in the required format)___end_action_input___
___end_response___

In the next iteration you will be given the Observation from the previous iteration.
You can then choose the tool to give the final response, or you can continue to use the other tools to gather more observations.

Begin!

Question: {input}
{agent_scratchpad}

Response format:
___start_response___
___start_thought___<mandatory>___end_thought___
___start_action___<mandatory>___end_action___
___start_action_input___<mandatory>___end_action_input___
___end_response___

Response:
"""

# Set up a prompt template
class ThreeUnderscorePromptTemplate(BaseChatPromptTemplate):
    # The list of tools available
    tools: List[Tool]
    
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
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tools"] += "\nFinalResponse: The final response to the original input question"
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["tool_names"] += ", FinalResponse"
        kwargs["tool_count"] = len(self.tools) + 1
        formatted = template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
