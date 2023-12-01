import os
import re

from typing import List, Union, Callable

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY


def fake_func(inp: str) -> str:
    return 'foo'


def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata['index']] for d in docs]


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ################### NEW ###################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop('intermediate_steps')
        thoughts = ''
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs['agent_scratchpad'] = thoughts
        ################### NEW ###################
        tools = self.tools_getter(kwargs['input'])
        # Create a tools variable from the list of tools provided
        kwargs['tools'] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools proviced
        kwargs['tool_names'] = ', '.join([tool.name for tool in tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={'output': llm_output.split("Final Answer:")[-1].strip()},
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


if __name__ == '__main__':
    # Define which tools the agent can use to answer user queries
    search = SerpAPIWrapper()
    search_tool = Tool(
        name='Search',
        func=search.run,
        description='useful for when you need to answer questions about current events',
    )

    fake_tools = [
        Tool(
            name=f"foo-{i}",
            func=fake_func,
            description=f"a silly function that you can use to get more information about the number {i}",
        )
        for i in range(99)
    ]

    ALL_TOOLS = [search_tool] + fake_tools

    docs = [Document(page_content=t.description, metadata={'index': i})
            for i, t in enumerate(ALL_TOOLS)]

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

    retriever = vector_store.as_retriever()

    # print(get_tools("whats the weather?"))
    # print(get_tools("whats the number 13?"))

    # Set up the base template
    template = """Answer the following questions as best you can, but speaking as a pirate might speak.
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
    
    Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s
    
    Question: {input}
    {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template=template,
        tools_getter=get_tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=['input', 'intermediate_steps'],
    )

    output_parser = CustomOutputParser()

    llm = OpenAI(temperature=0)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tools = ALL_TOOLS
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent_executor.run(input="What's the weather in SF?")
