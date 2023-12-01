import os
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
df = pd.read_csv('./basics/japan.csv')
agent = create_csv_agent(OpenAI(temperature=0.7,openai_api_key=OPENAI_API_KEY),
                         [],
                         verbose=True)
# print(agent.run("介绍下该数据集中包含的信息"))
# print(agent.run("从1960到2018年日本饮酒人口占比的走势是怎样的，介绍下"))
# print(agent.run("从1960到2018年日本80岁以上女性数量的走势是怎样的，介绍下"))
print(agent.run("生成对2000年到2018年日本婴儿死亡人数，可视化代码"))