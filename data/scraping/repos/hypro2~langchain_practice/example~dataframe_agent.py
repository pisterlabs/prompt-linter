import os
from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import pandas as pd
from langchain.llms import OpenAI

df = pd.read_csv("../dataset/mlb_teams_2012.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",openai_api_key=openai_api_key),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent.run("how many rows are there?")