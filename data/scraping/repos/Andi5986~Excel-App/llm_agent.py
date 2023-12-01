import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

def set_openai_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def init_agent(api_key, model_name='gpt-3.5-turbo', temperature=0.0):
    set_openai_key(api_key)
    ai_agent = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)
    return ai_agent

def get_agent_response(ai_agent, df, user_input):
    agent = create_pandas_dataframe_agent(ai_agent, df, verbose=True)
    response = agent.run(user_input)
    return response, df