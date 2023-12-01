from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from langchain.llms import OpenAI
import pandas as pd

import langchain
langchain.debug = True

df = pd.read_csv("titanic.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

print(agent.run("how many rows are there?"))