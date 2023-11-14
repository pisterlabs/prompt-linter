import re
import streamlit as st
from typing import List, Union, Callable

from langchain.prompts.base import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser, AgentExecutor, LLMSingleActionAgent
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config import MAX_ITERATIONS


class Custom_agent_exemple():
    '''This exemple can be found at https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval'''

     # If you make classes outside this one it will be considered as a class with a initialize_agent method
    def __init__(self):
        #Most of the init is useless here
        self.name='My custom Agent'
        self.llm = OpenAI(temperature=0,callbacks=[StreamingStdOutCallbackHandler()])
        self.tools = st.session_state.tools
        self.chat_history = st.session_state.chat_history
        self.memory = st.session_state.memory
        self.prefix=st.session_state.prefix
        self.suffix=st.session_state.suffix
        
        self.make_prompt()
        self.make_parser()
        

    def make_prompt(self):

        # Set up the base template
        template = self.prefix+ """
        
        Answer the question with very long and detailed explanations, be very precise and make clear points. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, can be one of {tool_names} is you need to use tool
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

       Here are your memories : {memory}

        Question: {input}
        {agent_scratchpad}
        """ + self.suffix


        self.prompt = self.CustomPromptTemplate(
                        template=template,
                        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                        # This includes the `intermediate_steps` variable because that is needed
                        input_variables=["input", "intermediate_steps"],
                
    )
        

    def make_parser(self):
        self.output_parser = self.CustomOutputParser()

   
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
            return AgentAction(
                tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
            )
    

    class CustomPromptTemplate(StringPromptTemplate):
        template : str
        tools_getter : Callable
        def get_tools(self,query):
            query_list_tool=st.session_state.tools
            return query_list_tool     
        def __init__(self,template='',**kwargs):
           
            super().__init__(template=template,tools_getter=self.get_tools,**kwargs)
            self.template=template
            self.tools_getter=self.get_tools
            
        
       
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
            kwargs["memory"] = st.session_state.memory
            
            tools = self.tools_getter(kwargs["input"])
            # Create a tools variable from the list of tools provided
            kwargs["tools"] = "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            )
            # Create a list of tool names for the tools provided
            kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
         
            return self.template.format(**kwargs)

    def initialize_agent(self):

  
            llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

            agent = LLMSingleActionAgent(
                    llm_chain=llm_chain,
                    output_parser=self.output_parser,
                    stop=["\Final answer:"],
                    allowed_tools=[t.name for t in self.tools],
                )
            
            agent_executor = AgentExecutor.from_agent_and_tools(
                        agent=agent, tools=self.tools, verbose=True,memory=self.memory,
                        handle_parsing_errors=True,
                        max_iterations=MAX_ITERATIONS,
                        streaming=True,
                )
            
            return agent_executor

     