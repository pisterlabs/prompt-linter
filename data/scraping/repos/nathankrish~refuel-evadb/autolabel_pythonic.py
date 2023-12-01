from autolabel import LabelingAgent, AutolabelDataset
import os
from my_secrets import OPENAI_KEY
# provide your own OpenAI API key here
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

agent = LabelingAgent(config='config_banking.json')

ds = AutolabelDataset('seed.csv', config = 'config_banking.json')
print(agent.plan(ds))

ds = agent.run(ds)
print(ds.df.head())