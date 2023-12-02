
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import SystemMessage
from langchain.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from .formula_tools import DebugFormulaCodeLLM, LoadFormulaCode, QueryFormulaCode, DecodeFormulaCodeLLM
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from .config import cfg
from .prompts import FIX_CODE_PREFIX, QUERY_PROMPT, SAMPLE_QUERY
from langchain.callbacks.base import BaseCallbackHandler

os.environ["OPENAI_API_KEY"] = cfg["OPENAI_API_KEY"]


if cfg["LANGCHAIN_API_KEY"] != "":
	os.environ["LANGCHAIN_TRACING_V2"] ="true"
	os.environ["LANGCHAIN_ENDPOINT"] ="https://api.smith.langchain.com"
	os.environ["LANGCHAIN_API_KEY"] = cfg["LANGCHAIN_API_KEY"]
	os.environ["LANGCHAIN_PROJECT"] ="pt-oily-sultan-99"


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")


system_message = SystemMessage(content=FIX_CODE_PREFIX)
_prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools = [LoadFormulaCode, QueryFormulaCode, DecodeFormulaCodeLLM(llm=llm), DebugFormulaCodeLLM(llm=llm)]

agent = OpenAIFunctionsAgent(
    llm=llm,
    prompt=_prompt,
    tools=tools,
    memory=memory,
    verbose=True
    )

agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

code = """\
domain Battery
{
  Component ::= new (label : String, weight : Real).
  Battery ::= new (label : String, weight : Real, capacity : Real).

  // Energy consumption rate
  rate ::= (Real).

  // Duration to complete the MissionItem with the given name
  itemDuration ::= (String, Real).

  // Amount of energy consumed to carry out the Mission with given name
  itemConsumption ::= (String, Real).

  // Total battery capacity
  batteryCapacity ::= (Real).

  // (x,y) location of the drone
  Loc ::= new (x : Real, y : Real).

  // Each mission item involves moving from source to destination locs at given velocity
  MissionItem ::= new (label : String, src : Loc, dest : Loc, dist : Real, vel : Real).

  // Each mission is a list of mission items
  Mission ::= new (m : MissionItem, remainder : any Mission + {NIL}).

  batteryCapacity(c) :- Battery(_, _, c).

  itemDuration(name, t) :- MissionItem(name, _, _, dist, vel), t = dist/vel.

  itemConsumption(name, c) :- MissionItem(name, _, _, _, _),
    itemDuration(name, t1),
    c = t1.
    insufficientBattery :- itemConsumption(_, x), batteryCapacity(c), x > c.
    conforms no insufficientBattery.
}

partial model pm of Battery
{
  Component("payload1", 5).
  Component("payload2", 3).
  Component("body", 10).

  // Battery capacity is symbolic
  Battery("battery1", 5, 10).

  l1 is Loc(40.00, 5.00).
  l2 is Loc(47.00, 8.00).
  l3 is Loc(52.00, 2.00).

  t1 is MissionItem("task1", Loc(40.00, 5.00), Loc(47.00, 8.00), 7.62, 0.04).
  t2 is MissionItem("task2", Loc(47.00, 8.00), Loc(52.00, 2.00), 7.81, m).

  m1 is Mission(t1, m2).
  m2 is Mission(t2, NIL).

  // rate = (225.4)/2.1 = 107.33
}
"""

interpreter_output = """ \
[]> solve pm 1 Battery.conforms
Parsing text took: 1
Visiting text took: 0
Started solve task with Id 0.
0.06s.
[]> ls

Environment variables

Programs in file root
 +-- /
 | tmp_file.4ml

Programs in env root
 +-- /

All tasks
 Id | Kind  | Status | Result |      Started      | Duration
----|-------|--------|--------|-------------------|----------
 0  | Solve |  Done  | false  | 7/14/2023 3:44 PM |  0.28s
0.02s.
[]> ex 0 1 out.4ml
Model not solvable.
0.02s.
"""

additional_details = """ \
The model is trying to model a battery in a drone nad figure out how far we can go on a missio.
It seems to be having trouble with the battery capacity, if we could help use some type of symbolic value for the battery capacity, that would be great.
"""

# agent_executor.run(QUERY_PROMPT.format(code=code, interpreter_output=interpreter_output, additional_details=additional_details))
if __name__ == "__main__":
	agent_executor.run(SAMPLE_QUERY)