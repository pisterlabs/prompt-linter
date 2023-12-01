#!/usr/bin/python3
import os
import json
import time
import cgi
print("Content-Type: text/html")
print()

form = cgi.FieldStorage()
data1=form.getvalue("search_query1")
print(data1)
from langchain.llms import OpenAI
my_openapi_key="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"
from langchain.chains import LLMChain


myllm = OpenAI(temperature=0, openai_api_key=my_openapi_key)


output = myllm( prompt= data1)

print(output)

from langchain.prompts import PromptTemplate

myprompt=PromptTemplate(
    template="tell me top 2 {thing} of india ,Give only name of it." ,
    input_variables=["thing"]
)

myprompt.format(thing="birds")
output = myllm( prompt=myprompt.format(thing="birds") )

my_things_prompt=myprompt.format(thing="birds")

type(myllm)

mychain= LLMChain(
    prompt=myprompt , 
    llm=myllm
)
data2=form.getvalue("search_query2")
print(mychain.run(thing=data2))

from langchain.agents import load_tools

import os

# Replace 'YOUR_API_KEY' with your actual SerpApi API key
api_key = 'c384209be726262c36ae60aeefad89ffe2e91ebb3ffa8ff904c1ca179cc93127'

# Set the environment variable
os.environ['SERPAPI_API_KEY'] = api_key

mytools= load_tools(tool_names = ["serpapi"] ,llm=myllm)

from langchain.agents import initialize_agent

from langchain.agents import AgentType

my_google_chain =initialize_agent(tools=mytools , 
                                  llm=myllm ,
                                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=True)
data3=form.getvalue("search_query3")
my_google_chain.run(data3)
