import re
from typing import List, Union

from langchain.prompts import PromptTemplate, BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.get("intermediate_steps")
        thoughts = ""
        if intermediate_steps!=None:
          for action, observation in intermediate_steps:
              thoughts += action.log
              thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n\n".join([f"{tool.name}: \n{tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        return_values = {"output": {"tool_used": "", "result": ""}}
        if "Action:" in llm_output:
            return_values["output"]["tool_used"] = llm_output.split("Action:")[-1].strip().split("\n")[0]
        if "Action Input:" in llm_output:
            return_values["output"]["result"] = llm_output.split("Action Input:")[-1].strip()
        if(len(return_values["output"].items())==2):
          return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values = return_values,
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input

        ## write regex code to store the observation of the conversation in the memory
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)