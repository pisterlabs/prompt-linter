import json

from .agents import defined_agent, agent_type
from .agent_input import Agent_Input
from .multi_agent_prompts import query_type_agent_prompt

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

#Helper classes

class query_type_agent_input(Agent_Input):
    def __init__(self, prompt_to_analyse: str) -> None:
        super().__init__()
        self.prompt_to_analyse = prompt_to_analyse

class query_type_agent_types(agent_type):
    RAG_QUERY_TYPE = "RAG_QUERY_TYPE"

#Main class

class query_type_agent(defined_agent):
    def __init__(self, llm):
        self.type = type
        super().__init__(query_type_agent_types.RAG_QUERY_TYPE, llm, query_type_agent_prompt)

    def run(self, input: query_type_agent_input):
        promptToAnalyse = PromptTemplate(input_variables=["query"], template=self.prompt)

        chain = LLMChain(llm=self.llm, prompt=promptToAnalyse)
        return chain.run(query=input.prompt_to_analyse)

    async def arun(self, input: query_type_agent_input):
        promptToAnalyse = PromptTemplate(input_variables=["query"], template=self.prompt)

        chain = LLMChain(llm=self.llm, prompt=promptToAnalyse)
        return chain.arun(query=input.prompt_to_analyse)