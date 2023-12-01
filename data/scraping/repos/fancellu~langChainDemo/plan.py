from langchain import LLMMathChain
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner

if __name__ == '__main__':
    prompt = "Where are the next summer olympics going to be hosted? What is the population of that country, squared?"

    llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    search = DuckDuckGoSearchAPIWrapper()

    tools = [
        Tool(name="Search", func=search.run, description="For searching about current events"),
        Tool(name="Calculator", func=llm_math_chain.run, description="For maths"),
    ]
    # ChatOpenAI has a memory, useful!
    model = ChatOpenAI(temperature=0)
    planner = load_chat_planner(model)
    executor = load_agent_executor(model, tools, verbose=True)

    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

    agent.run(prompt)
