from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])

llm = OpenAI(model="text-davinci-003", temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("What is the meaning of life, universe and everything?"))