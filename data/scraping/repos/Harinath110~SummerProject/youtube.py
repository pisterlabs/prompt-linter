#!/usr/bin/python3
import os
import json
import time
import cgi
print("Content-type: text/html")
print()

form=cgi.FieldStorage()
xz=form.getvalue("name3")

from langchain.llms import OpenAI
from langchain.tools import YouTubeSearchTool
tool = YouTubeSearchTool()
x=tool.run(xz)
print(x)
