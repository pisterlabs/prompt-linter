import json

from .agents import defined_agent, agent_type
from .agent_input import Agent_Input
from .multi_agent_prompts import rag_agent_prompt

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class rag_prompt_agent_input(Agent_Input):
    def __init__(self, prompt_to_analyse: str) -> None:
        super().__init__()
        self.prompt_to_analyse = prompt_to_analyse

class rag_prompt_agent_types(agent_type):
    RAG_PROMPT_TYPE = "RAG_PROMPT_TYPE"

class rag_prompt_agent(defined_agent):
    def __init__(self, llm):
        self.type = type
        super().__init__(rag_prompt_agent_types.RAG_PROMPT_TYPE, llm, rag_agent_prompt)

    def run(self, input: rag_prompt_agent_input):
        promptToAnalyse = PromptTemplate(input_variables=["query"], template=self.prompt)

        chain = LLMChain(llm=self.llm, prompt=promptToAnalyse)
        return chain.run(query=input.prompt_to_analyse)

    async def arun(self, input: rag_prompt_agent_input):
        promptToAnalyse = PromptTemplate(input_variables=["query"], template=self.prompt)

        chain = LLMChain(llm=self.llm, prompt=promptToAnalyse)
        return chain.arun(query=input.prompt_to_analyse)