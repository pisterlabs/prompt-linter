'''
pip install openai
pip install langchain
'''

import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI()
answer = llm("Why did the chicken cross the road?")
print(answer)
