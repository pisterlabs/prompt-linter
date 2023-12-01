#!/usr/bin/python3


import subprocess
import cgi
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

print("Content-Type: text/html")
print("Access-Control-Allow-Origin: *")
print("Access-Control-Allow-Methods: POST, GET, OPTIONS")
print("Access-Control-Allow-Headers: Content-Type")
print()

formdata = cgi.FieldStorage()
prompt = formdata.getvalue("message")
# cmd = "date"
# output = subprocess.getoutput(cmd)
# print(output)
myOpenAPiKey = "YOUR_OPENAI_API_KEY"
myLLM = OpenAI(
    temperature=0.5,
    max_tokens=1000,
    openai_api_key=myOpenAPiKey
)
template = """  you are an 10 years experienced programmer ,
so I will be giving you some coding  questions.
So write a code for {code}.
1. Explain each line of code in detail. """
# print('test1')
myprompttemplate = PromptTemplate(
    input_variables=['code'],
    template=template
)
mychain = LLMChain(
    llm=myLLM,
    prompt=myprompttemplate
)

agent_executor = create_python_agent(

    llm=myLLM,
    tool=PythonREPLTool(),
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False

)
# print('test2')
# output = agent_executor.run("Write a code for fibonacci sequence ")
output = agent_executor.run(prompt)
# print('test3')
print("<pre>")
print("<h3>")
print(output)
print("</h3>")
print("</pre>")
