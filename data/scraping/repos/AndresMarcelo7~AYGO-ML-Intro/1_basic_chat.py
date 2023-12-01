# get a token: https://platform.openai.com/account/api-keys

import os
from getpass import getpass
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

#OPENAI_API_KEY = getpass()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

response = llm_chain.run(question)

print(response)
