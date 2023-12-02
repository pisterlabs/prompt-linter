import os
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd



OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')


df = pd.read_csv('samples/sales_data_sample.csv')
df[0:10]

pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

pd_agent.run("Total sales")
