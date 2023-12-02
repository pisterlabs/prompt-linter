import os

import openai
import json
from dotenv import load_dotenv
from AutoGPT import WriteFile, ReadFile

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

tools = [
    WriteFile()
]

functions = [tool.tool_to_function() for tool in tools]

messages = [
    {
        "role": "user",
        "content": "Write ABC to sample.txt"
    }
]

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7,
                                          functions=functions, function_call="auto")
reply = completion.choices[0].message
print(reply)

if reply.function_call:
    for tool in tools:
        if reply.function_call["name"] == tool.name:
            tool.run(json.loads(reply.function_call["arguments"]))
