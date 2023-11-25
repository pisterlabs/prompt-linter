import environment

from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

import os

# from getpass import getpass
# HUGGINGFACE_API_KEY = getpass()

from langchain.llms import Petals
from langchain import PromptTemplate, LLMChain
llm = Petals(model_name="bigscience/bloom-petals")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))