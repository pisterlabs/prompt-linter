from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from util import initialize

initialize()

model = ChatOpenAI(temperature=0)

tools = load_tools(["llm-math", "ddg-search"], llm=model)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
res = agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
)
print(res)
