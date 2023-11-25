import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

import os
from langchain.llms import GooseAI
from langchain import PromptTemplate, LLMChain

# from getpass import getpass
# GOOSEAI_API_KEY = getpass()
# os.environ["GOOSEAI_API_KEY"] = GOOSEAI_API_KEY

llm = GooseAI()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))