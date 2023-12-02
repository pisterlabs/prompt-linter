from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(model="text-davinci-003", temperature=0.7)

template = "What is a good name for a company that makes {product} product?"
input_variables = re.findall(r"\{([^}]+)\}", template)
prompt = PromptTemplate(template=template, input_variables=input_variables)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("wireless headphones"))