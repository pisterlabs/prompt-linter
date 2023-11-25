#pip install langchain
#pip install openai

import os
from langchain.llms import OpenAI
my_openapi_key="Your_api_key"
from langchain.chains import LLMChain


myllm = OpenAI(temperature=0, openai_api_key=my_openapi_key)
#0: no random 0.5 1.0

output = myllm( prompt= "tell me top 2 {x} of india ,Give only name of it.x=food")

print(output)

from langchain.prompts import PromptTemplate

myprompt=PromptTemplate(
    template="tell me top 2 {things} of india ,Give only name of it." ,
    input_variables=["things"]
)

myprompt.format(things="animals")
output = myllm( prompt=myprompt.format(things="animals") )

my_things_prompt=myprompt.format(things="animals")

type(myllm)

mychain= LLMChain(
    prompt=myprompt , 
    llm=myllm
)

print(mychain.run(things="food"))

from langchain.agents import load_tools

import os

# Replace 'YOUR_API_KEY' with your actual SerpApi API key
api_key = 'your_api_key'

# Set the environment variable
os.environ['SERPAPI_API_KEY'] = api_key

 mytools= load_tools(tool_names = ["serpapi"] ,llm=myllm)

from langchain.agents import initialize_agent

from langchain.agents import AgentType

my_google_chain =initialize_agent(tools=mytools , 
                                  llm=myllm ,
                                  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 verbose=True)

my_google_chain.run("tell me the current president of US in one lines")

import os

os.environ["WOLFRAM_ALPHA_APPID"] = "6PVJ2L-3WK3UAPW58"

from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

wolfram = WolframAlphaAPIWrapper()

wolfram.run("What is 2x+5 = -3x + 7?")

a=wolfram.run("What is 8x+2 = -9x + 8?")

a

