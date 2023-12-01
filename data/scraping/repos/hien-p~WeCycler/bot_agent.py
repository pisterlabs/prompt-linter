from langchain.llms import BaseLLM
from langchain.agents import AgentType, initialize_agent

import sys
sys.path.append('../')
from botcore.chains.assess_usage import build_assess_usage_tool
from botcore.chains.pros_cons import build_pros_cons_tool
from botcore.chains.recycling_tip import build_recycling_tip_tool

class AgentBot:

    def __init__(self, model: BaseLLM, memory):
        tools = self.load_tools(model, memory)
        self.agent = initialize_agent(tools, model,\
                                      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,\
                                      verbose=True, return_intermediate_steps=True)
        print("Agent is ready")
    
    def answer(self, question: str, return_only_output: bool = True):
        resp = self.agent(question)
        if return_only_output:
            return resp['output'] # str

        return resp
        
    def load_tools(self, model, memory):
        tools = [build_assess_usage_tool(model,memory),
                 build_pros_cons_tool(model,memory), build_recycling_tip_tool(model, memory)]
        return tools

    
