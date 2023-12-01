import os
from constants import openai_key
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"]=openai_key


## OPENAI LLMS
llm=OpenAI(temperature=0.8)


text =  input("enter the topic you want:  ")
result = llm(text)
print(result)