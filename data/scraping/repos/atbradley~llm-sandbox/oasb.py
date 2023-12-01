import os

import json
import yaml
import time

import openai
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4") 
openai.api_key = os.getenv("API_KEY")

# TODO: New version that uses Chat Completion API. and sends a list of messages
# with each request: https://platform.openai.com/docs/guides/gpt/chat-completions-api

while True:
    prompt = input("Prompt ('q' to quit, 'f' to read from file.): ")
    if prompt == "q":
        break

    if prompt == "f":
        #TODO: Offer to default to the newest .md file.
        try:
            with open(input('Filename: '), "r") as f:
                prompt = f.read()
        except FileNotFoundError:
            print("File not found.")
            continue
    
    print("sending...")
    response = openai.ChatCompletion.create(model=MODEL, 
                    messages=[{"role": "user", "content": prompt}])
    
    fname = f"response{time.time_ns()}.json"
    transdata = {
        "request": {
            "prompt": prompt,
        },
        "response": response,
    }
    jsondata = json.dumps(transdata, indent=4)
    with open(fname, "w") as f:
        f.write(jsondata)

    fname = f"response{time.time_ns()}.yaml"
    yamldata = json.loads(jsondata)
    yaml.dump(yamldata, open(fname, "w"), indent=4, Dumper=yaml.SafeDumper)

    print()
    #print(response.keys())
    message = response['choices'][0]['message']
    print("role:", message.get('role', "None"))
    print("content:", message.get('content', "No content"))
    print()
    print()
