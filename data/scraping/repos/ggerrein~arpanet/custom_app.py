from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

import re
import langchain
import importlib
import sys
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI



# Set up the base template
template = """You are a business analyst and understand how all the different company data sources fit together  
If a user asks for "hits" means "page views" or "honey mustards". 
If a user asks for total hits or page views, you have to sum(hits) for that time period.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! 

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

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
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print(f"llm_output> {llm_output}")


        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip(),"llm_output":llm_output},
                log=llm_output,
            )
        # Parse out the action and action input

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def get_app(tools,str):
        prompt = CustomPromptTemplate(
            template=template,
            tools=tools,
            input_variables=["input", "intermediate_steps"]
        )

        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")
        agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in tools],
        )

        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

        return agent_executor([str])