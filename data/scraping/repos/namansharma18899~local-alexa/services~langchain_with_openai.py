import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = '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'


class Langchain_OpenAI:
    def __init__(self) -> None:
        pass

    def run_agent(self, question):
        # TODO replace it with some locally hosted llm 
        llm = OpenAI(temperature=0)
        # perform calculation, search things from wikipedia and run commands in te
        #  terminal 
        # TODO custom tool need to be added based on need
        tools = load_tools(["llm-math","wikipedia","terminal"], llm=llm)
        # TODO use appropriate
        agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)
        print(agent.agent.llm_chain.prompt.template)
        return agent.run(question)
# driver 
# lc = Langchain_OpenAI()
# print(lc.run_agent("What is indian recipe of okra?"))
    