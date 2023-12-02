# get a token: https://huggingface.co/docs/api-inference/quicktour#get-your-api-token
# persist it to your HUGGINGFACEHUB_API_TOKEN environment variable

import os

from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType, load_huggingface_tool

HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"] 


repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

hugging_tool = load_huggingface_tool("lysandre/hf-model-downloads")

agent = initialize_agent([hugging_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
question = "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"

print(agent.run(question))