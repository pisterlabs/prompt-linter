""" 
Created on : 02/05/23 4:27 pm
@author : ds  
"""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from config import Open_AI
from langchain.chains import LLMChain

llm = OpenAI(openai_api_key=Open_AI.key)
# text = "What would be a good company name for a company that makes colorful socks?"
# print(llm(text))

prompt = PromptTemplate(input_variables=["product"],
                        template="What would be a good company name for a company that produce {product}")
chain = LLMChain(llm=llm, prompt=prompt)
print(LLMChain(llm=llm, prompt=prompt).run("whisky"))
