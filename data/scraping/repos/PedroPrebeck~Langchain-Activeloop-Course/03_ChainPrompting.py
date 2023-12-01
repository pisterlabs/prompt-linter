from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

template_question = """
What is the name of the famous scientist who developed the theory of general relativity?
"""
prompt_question = PromptTemplate(template=template_question, input_variables=[])

template_fact = """
Provide a brief description of {scientist}'s theory of general relativity.
Answer: 
"""
prompt_fact = PromptTemplate(template=template_fact, input_variables=["scientist"])

chain_question = LLMChain(llm=llm, prompt=prompt_question)
response_question = chain_question.run({})

scientist = response_question.strip()

chain_fact = LLMChain(llm=llm, prompt=prompt_fact)
input_data = {"scientist": scientist}

response_fact = chain_fact.run(input_data)

print(f"Scientist: {scientist}")
print(f"Fact: {response_fact}")