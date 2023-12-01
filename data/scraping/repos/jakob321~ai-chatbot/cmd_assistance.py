#!/usr/bin/python
import openai
import json
import os
import sys
from dotenv import dotenv_values

# read key from .env file
script_directory = os.path.dirname(os.path.abspath(__file__))
openai.api_key = dotenv_values(f"{script_directory}/.env")["OPENAI_API_KEY"]

linux_command_function = [
            {
                "name": "run_linux_commands",
                "description": "run commands in linux terminal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "command to run",
                        },
                    },
                    "required": ["command"],
                },
            }
        ]

# dummy data for not wasting api calls
dummy_data = {
  "id": "chatcmpl-7czAV6O0MduiNq8UhghYPYvxANGP4",
  "object": "chat.completion",
  "created": 1689525951,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "null",
        "function_call": {
          "name": "run_linux_commands",
          "arguments": "{\n  \"command\": \"ls -l\"\n}"
        }
      },
      "finish_reason": "function_call"
    }
  ],
  "usage": {
    "prompt_tokens": 55,
    "completion_tokens": 22,
    "total_tokens": 77
  }
}

def call_chatgpt(query, func):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": query},
        ],
        functions = func
    )
    return completion

def got_function_call(response):
  if(response["choices"][0]["message"].get("function_call")):
    return True
  return False

def get_function_arg(response, arg):
  arguments = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
  return arguments[arg]

def run_linux_commands(command, sudo=False):
  if(sudo):
    command = "sudo " + command
  print("running command: ", command)
  if(input("confirm? (y/n) ") == "y"):
    os.system(command)

if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    response = call_chatgpt(query, linux_command_function)
    if(got_function_call(response)):
      command = get_function_arg(response, "command")
      run_linux_commands(command, True)
    else:
      print(response["choices"][0]["message"]["content"])