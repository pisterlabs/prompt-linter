from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

prompt_question = PromptTemplate(template="""
What is the name of the famous scientist who developed the theory of relativity?

Answer:
""", input_variables=[])


prompt_question_moreinfo = PromptTemplate(template="""
Provide a brief description of {scientist}'s theory of relativity.

Answer:
""", input_variables=["scientist"])

chain_question = LLMChain(llm=llm, prompt=prompt_question)
response_question = chain_question.run({})
scientist = response_question.strip()

chain_question_moreinfo = LLMChain(llm=llm, prompt=prompt_question_moreinfo)
response_question_moreinfo = chain_question_moreinfo.run({"scientist": scientist})

print("Scientist: ", scientist)
print("Theory of relativity: ", response_question_moreinfo)