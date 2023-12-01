from dotenv import load_dotenv
import re

variable_pattern = r"\{([^}]+)\}"

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

template = "Question: {question}\nAnswer:"
input_variables = re.findall(variable_pattern, template)

prompt = PromptTemplate(template=template, input_variables=input_variables)

llm = OpenAI(model="text-davinci-003", temperature=0.7)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("What is the meaning of life?"))