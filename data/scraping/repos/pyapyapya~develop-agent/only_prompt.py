import openai
from dotenv import load_dotenv
import os
import json
from typing import TypedDict

load_dotenv()

class DotEnv(TypedDict):
    OPEN_API_KEY: str

settings: DotEnv = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
}

openai.api_key = settings["OPENAI_API_KEY"]

def get_prompt(query: str):
    prompt = \
    """
    Consider the following sentences:
    Given Question
    ------
    "%s"

    ------

    """
    return prompt % query

prompt = \
"""
Request: Please develop a website that displays hello world
You have python in shell. Don't use another program, for example python.

Process
--------
-  If not installed flask package, install flask and use it.
-  [Request] using flask package, develop a website that displays hello world
-  Add code using 'echo' command to /home/taehyeon/develop-agent/develop-agent/page.py
-  Execute server when connect to http://localhost:8000, then user can see "hello world"
--------
"""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)
data = json.loads(response.choices[0].text)