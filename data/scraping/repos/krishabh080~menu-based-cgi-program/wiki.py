#!/usr/bin/python3

import cgi 
import time
import threading
import langchain
import openai
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
print("Content-type:text/html")
print()

form = cgi.FieldStorage()
data=form.getvalue("c")
def run(data):
 wikipedia = WikipediaAPIWrapper()
 myapikey= openai.api_key="your_openai_key"
 myllm = OpenAI(
    model= ('text-davinci-003'),
    temperature=1,
    openai_api_key= myapikey
    )
 mywikitool=load_tools(tool_names=["wikipedia"])
 mywikichain=initialize_agent(
 llm=myllm,
 tools=mywikitool,
 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
 verbose=True

 )


 mywikichain.run(data)

program_thread = threading.Thread(target=run, args=(data))
program_thread.start()
program_thread.join()
op=run(data)
#time.sleep(30)

print(op)
print()
print()
print("<form action= HTTP://'your_ip'/menu.html>")
print("<input type='submit' vaue='Back to Main menu'></form>")

