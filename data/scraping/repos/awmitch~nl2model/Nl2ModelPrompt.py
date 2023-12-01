from langchain.prompts import StringPromptTemplate
from langchain.tools import BaseTool
from langchain.memory import ConversationSummaryMemory,ConversationBufferMemory
from typing import List

class Nl2ModelPrompt(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]
    memory: ConversationSummaryMemory

    def format(self, **kwargs) -> str:
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
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["chat_history"] = f"{self.memory.buffer}"
        formatted = self.template.format(**kwargs)
        
        return formatted

