import os

from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import OpenAI, SerpAPIWrapper, LLMMathChain
from langchain.agents import Tool

import config as c

os.environ['OPENAI_API_KEY'] = c.OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = c.SERPAPI_API_KEY
os.environ['SERPER_API_KEY'] = c.SERPER_API_KEY
os.environ['GOOGLE_API_KEY'] = c.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = c.GOOGLE_CSE_ID


if __name__ == '__main__':
    # Tools
    search_params = {
        "engine": "google",
        "google_domain": "google.com",
        "gl": "kr",
        "hl": "en",
    }
    search = SerpAPIWrapper(params=search_params)
    llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        Tool(
            name='Search',
            func=search.run,
            description='useful for when you need to answer questions about current events',
        ),
        Tool(
            name='Calculator',
            func=llm_math_chain.run,
            description='useful for when you need to answer questions about math',
        ),
    ]

    # Planner, Executor, and Agent
    model = ChatOpenAI(temperature=0)
    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)
    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)  # 얘도 결국 chain

    # Run Example
    # print(agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"))
    print(agent.run("Please tell me the operating profit of Samsung Electronics in 2023"))



