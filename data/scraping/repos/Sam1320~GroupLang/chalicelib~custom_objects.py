import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain import LLMChain
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, Tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute.schema import (Plan,
                                                            PlanOutputParser,
                                                            Step,
                                                            ListStepContainer
                                                            )
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from telegram import Bot


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
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
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
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
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    

class Query:
    """A query from an user."""
    def __init__(self, client_id, client_username, query_text, last_action, moderator_id):
        self.client_id = client_id # id of the client who asked the question, also the expert on the meaning of the question
        self.client_username = client_username
        self.query_text = query_text
        self.last_action = last_action
        self.intermediate_steps = []
        self.moderator_id = moderator_id # expert on the reasoning of the bot
        self.expert_id = None # expert on the currently required piece of information
        self.expert_username = None
        self.final_answer = None
        self.needs_feedback = False # whether the bot needs feedback on its reasoning


class Task:
    """A task for an agent to complete."""
    def __init__(self, client_id, objective, plan):
        self.client_id = client_id
        self.objective = objective
        self.plan = plan
        self.current_step_n = 0
        self.step_container = ListStepContainer()
        self.last_action = None
        self.intermediate_steps = []

    def __str__(self):
        text = f"Client id: {self.client_id}\n"
        text += f"Objective: {self.objective}\n"
        text += "Plan:\n"
        for i, step in enumerate(self.plan.steps):
            text += f'{i}. {step.value}\n'
        text += "\n"
        # current step
        text += f"Current step:\n {self.current_step_n}. {self.plan.steps[int(self.current_step_n)]}\n"

        # previous steps
        text += "Previous steps:\n"
        for i, step in enumerate(self.step_container.steps):
            text += f'{i}.\n value: {step[0].value}\n  response: {step[1].response}\n'

        text += "\n"
        # intermediate steps
        text += "Intermediate steps:\n"
        for i, step in enumerate(self.intermediate_steps):
            text += f'{i}. action: {step[0].log}\n observation: {step[1]}\n'
        
        text += "\n"
        # last action
        text += f"Last action: {self.last_action}\n"
        return text
        


        

class PlanningOutputParser(PlanOutputParser):
    def parse(self, text: str) -> Plan:
        steps = [Step(value=v) for v in re.split("\n\d+\. ", text)[1:]]
        return Plan(steps=steps)