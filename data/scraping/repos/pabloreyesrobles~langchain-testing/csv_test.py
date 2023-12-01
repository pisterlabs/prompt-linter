import openai

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.agents import create_csv_agent
import constants
import os

os.environ["OPENAI_API_KEY"] = constants.APIKEY

agent = create_csv_agent(
    ChatOpenAI(model="gpt-3.5-turbo"),
    "data/data_entel.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run('Cuántos IMEI distintos hay en el dataset?')