import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain

search = SerpAPIWrapper()
# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]


# from langchain.chat_models import ChatOpenAI
# model = ChatOpenAI(temperature=0)
model = llm
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executer=executor, verbose=True)

agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")

