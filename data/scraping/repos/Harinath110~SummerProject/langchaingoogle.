#!/usr/bin/python3

print("content-type: text/html")
print()

import cgi
data=cgi.FieldStorage()
promp=data.getvalue("name")

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.agents import initialize_agent
import os


myllm=OpenAI(
    model="text-davinci-003",
    openai_api_key="...",
    temperature=0
    )
mylw=PromptTemplate(
    template="tell me two best{item} in {country}. ",
    input_variables=["item","country"]    

    )
mychain=LLMChain(
    llm=myllm,
    prompt=mylw
)
os.environ['SERPAPI_API_KEY']="..."

myserptool=load_tools(tool_names=['serpapi'])

mygooglechain=initialize_agent(
    llm=myllm,
    tools=myserptool,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True

)

mygooglechain.run(promp)
