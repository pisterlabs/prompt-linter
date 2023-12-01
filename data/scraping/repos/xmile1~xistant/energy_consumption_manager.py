import os
from typing import Any
from click import prompt
from langchain.tools import BaseTool
from langchain.agents.tools import Tool
from langchain.agents import initialize_agent
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner


class EnergyConsumptionPlugin():
  def __init__(self, model):
      self.model = model
  def get_lang_chain_tool(self):
     return [EnergyConsumptionPluginTool(), EnergyConsumptionRetrieverPluginTool(model=self.model)]
      

class EnergyConsumptionRetrieverPluginTool(BaseTool):
  name = "EnergyConsumption retriever"
  description = (
    "The tool answers questions about energy/electricity consumption, input should be the question"
  )
  model: Any
  return_direct = True

  def _run(self, query: str) -> str:
    f = open(os.path.join("data/energy_comsumption.txt"), "r")
    data = f.read()

    SYSTEM_PROMPT = (
    "You are an Energy Consumption Expert and you have is data:"
    "DATA:"
    f"{data}"
    "if any data is missing, have a plan to generate it"
    "for question relating to a month, if the data for the first or thirtieth or thirty first of that month is not available, create a plan to generate it"
    ""
    "Let's first understand the problem and devise a plan to solve the problem."
    " Please output the plan starting with the header 'Plan:' "
    "and then followed by a numbered list of steps. "
    "Please make the plan the minimum number of steps required "
    "to accurately complete the task. If the task is a question, "
    "the final step should almost always be 'Given the above steps taken, "
    "please respond to the users original question'. "
    "At the end of your plan, say '<END_OF_PLAN>'"
    )
 
    planner = load_chat_planner(self.model, SYSTEM_PROMPT)

    tools = [
      Tool(
        name="Missing data generator",
        description="generate missing energy consumption data, input should be the date of the missing data and data from before and after the missing data",
        func=self.generate_energy_data,
      ),
      Tool(
        name="Energy data retriever",
        description="retrieve energy consumption data",
        func=lambda date: data,
      ),


    ]

    executor = load_agent_executor(self.model, tools, verbose=True)

    agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
    return agent.run(query)

  async def _arun(self, query: str) -> str:
    """Use the tool asynchronously."""
    raise NotImplementedError("This tool does not support async")
 
  def generate_energy_data(self, date):
    # generate the data for the given date using linear interpolation
    return '01.06.2023,13696'

  
class EnergyConsumptionPluginTool(BaseTool):
  name = "EnergyConsumption saver"
  description = (
    "Save energy consumption reading to a vector store when a user request to save his energy data, input should be the reading value in kilwatt and the date, for example 31.05.2023,13696. if the date is not provided by the user, provide today's date"
  )
  return_direct = True
  def _run(self, query: str) -> str: 
    with open(os.path.join("data/energy_comsumption.txt"), "a") as f:
      # if the given date, check if the data for the first of that month is available, if not, generate it using linear interpolation

      f.write(f"\n{query}")
    return "Energy consumption reading saved"

  async def _arun(self, query: str) -> str:
      """Use the tool asynchronously."""
      raise NotImplementedError("This tool does not support async")
