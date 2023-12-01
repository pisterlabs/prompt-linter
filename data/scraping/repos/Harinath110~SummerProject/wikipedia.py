#!/usr/bin/python3
import os
import json
import time
import cgi
print("Content-type: text/html")
print()

from langchain.llms import OpenAI


form=cgi.FieldStorage()
ko=form.getvalue("name2")

from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

z=wikipedia.run(ko)
print(z)
