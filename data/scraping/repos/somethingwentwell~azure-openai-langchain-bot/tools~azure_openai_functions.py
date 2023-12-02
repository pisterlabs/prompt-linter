import json
from langchain.agents import Tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY") 
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")

class DocsInput(BaseModel):
    question: str = Field()

function_loaded = False

try:
    with open("./tools/openai_functions.json", "r") as f:
        functions = json.load(f)
        function_loaded = True
except Exception as e:
    print(f"Error: {e}")
    function_loaded = False

def json_output(message):
    if not function_loaded:
        return {"content": "Error: Functions not loaded. Please contact the administrator.",
                "total_tokens": 0}
    
    messages = [{
        "role": "user",
        "content": message
    }]
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=messages,
        functions=functions,
        function_call="auto", 
    )
    obj = response

    if (response["choices"]):
        obj = {
            "content": response["choices"][0]["message"]["function_call"]["arguments"],
            "total_tokens": response["usage"]["total_tokens"]
        }
    print(obj)

    return obj

def direct_json():
    tools = []
    tools.append(Tool(
        name = "JSON Formatter",
        func=json_output,
        description="Useful for when you need to answer questions using OpenAI. Input must be the exact same text as user's ask.",
        return_direct=True
    ))
    return tools


def azure_openai_functions():
    tools = []
    tools.extend(direct_json())
    return tools