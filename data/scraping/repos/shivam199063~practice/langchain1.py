
from langchain.llms import OpenAI
my_openai_key = "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"
mymodel = OpenAI(temperature=0 ,openai_api_key=my_openai_key)


from langchain import PromptTemplate

myprompt = PromptTemplate(
    template= "tell me top 2 {things} of india, which me only name of it.",
    input_variables=["things"]
)

# myprompt.format(things="food")

# output = mymodel( prompt=myprompt.format(things="animal"))

# print(output)


from langchain.chains import LLMChain


mychain = LLMChain(
    prompt=myprompt,
    llm=mymodel
)

# print(mychain.run(things="food"))


# ***********************************************************************************************

import os
os.environ["OPENAI_API_KEY"] = '50898974cf5534c59add384df3a25fc0c7927c747952394f3e93a74da47817f0'

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools

mytool = load_tools(tool_names=['serpapi'])



mygoogle_chain = initialize_agent(
    tools=mytool,
    llm = mymodel,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
    )

mygoogle_chain.run("who is vimal in two points")


