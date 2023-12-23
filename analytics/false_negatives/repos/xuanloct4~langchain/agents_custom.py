
import environment
from agents_tools import search_tool_serpapi

from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import SerpAPIWrapper
tools = [search_tool_serpapi()]
tool_names = [tool.name for tool in tools]

from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish

from langchain.memory import ConversationBufferWindowMemory
memory=ConversationBufferWindowMemory(k=2)


##FakeAgent
class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""
    
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")
fakeAgent = FakeAgent()




from llms import defaultLLM as llm
# Custom LLM Agent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

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



# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

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

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""


# Set up the base template
template_with_history = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

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

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Previous conversation history:
{history}

New question: {input}
{agent_scratchpad}"""

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
output_parser = CustomOutputParser()
llmSingleActionAgentWithMemory = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)


# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
llmSingleActionAgent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)


# Custom MRKL Agent
prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

from langchain.agents import ZeroShotAgent
mrklPrompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)
print(mrklPrompt.template)
mrklAgent = ZeroShotAgent(llm_chain=LLMChain(llm=llm, prompt=mrklPrompt), allowed_tools=tool_names)


multiInputPrefix = """Answer the following questions as best you can. You have access to the following tools:"""
multiInputSuffix = """When answering, you MUST speak in the following language: {language}.

Question: {input}
{agent_scratchpad}"""

multiInputPrompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=multiInputPrefix, 
    suffix=multiInputSuffix, 
    input_variables=["input", "language", "agent_scratchpad"]
)
# print(multiInputPrompt.template)
multiInputMRKLAgent = ZeroShotAgent(llm_chain=LLMChain(llm=llm, prompt=multiInputPrompt), tools=tools)

# agent_executor = AgentExecutor.from_agent_and_tools(agent=llmSingleActionAgent, tools=tools, verbose=True)
# agent_executor.run("How many people live in canada as of 2023?")

# agent_executor = AgentExecutor.from_agent_and_tools(agent=llmSingleActionAgentWithMemory, tools=tools, verbose=True, memory=memory)
# agent_executor.run("How many people live in canada as of 2023?")
# agent_executor.run("how about in mexico?")

# agent_executor = AgentExecutor.from_agent_and_tools(agent=fakeAgent, tools=tools, verbose=True)
# agent_executor.run("Search for Leo DiCaprio's girlfriend on the internet.")

# agent_executor = AgentExecutor.from_agent_and_tools(agent=mrklAgent, tools=tools, verbose=True)
# agent_executor.run("How many people live in canada as of 2023?")

agent_executor = AgentExecutor.from_agent_and_tools(agent=multiInputMRKLAgent, tools=tools, verbose=True)
agent_executor.run(input="How many people live in canada as of 2023?", language="italian")