import os
from langchain.agents import Tool
from dotenv import load_dotenv
import aiohttp
import requests
import json

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
load_dotenv(os.path.join(BASEDIR, '.env'), override=True)

def aoai(question):
        try:
            url = f"{str(os.getenv('OPENAI_API_BASE'))}/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview"
            payload = json.dumps({
            "messages": [
                {
                "role": "user",
                "content": question
                }
            ]
            })

            headers = {
            'api-key': str(os.getenv("OPENAI_API_KEY")),
            'Content-Type': 'application/json'
            }

            response = requests.post(url, headers=headers, data=payload)
            response_json = response.json()
            print(response_json)
            obj = {
                  "content": response_json["choices"][0]["message"]["content"],
                  "total_tokens": response_json["usage"]["total_tokens"]
            }
            return obj
                    
        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"


async def async_aoai(question):
        try:
            url = f"{str(os.getenv('OPENAI_API_BASE'))}/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-03-15-preview"

            payload = json.dumps({
            "messages": [
                {
                "role": "user",
                "content": question
                }
            ]
            })

            headers = {
            'api-key': str(os.getenv("OPENAI_API_KEY")),
            'Content-Type': 'application/json'
            }

            async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=payload) as response:
                            response_json = await response.json()
                            obj = {
                                "content": response_json["choices"][0]["message"]["content"],
                                "total_tokens": response_json["usage"]["total_tokens"]
                            }
                            return obj

        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"



def AOAI():
    tools = []
    tools.append(Tool(
        name = "Direct call Azure OpenAI",
        func=aoai,
        description="Useful for when you need to answer questions using OpenAI. Input must be the exact same text as user's ask.",
    ))
    return tools

def AAOAI():
    tools = []
    tools.append(Tool(
        name = "Direct call Azure OpenAI",
        func=aoai,
        description="Useful for when you need to answer questions using OpenAI. Input must be the exact same text as user's ask.",
        coroutine=async_aoai,
    ))
    return tools

def direct_gpt():
    tools = []
    tools.extend(AAOAI())
    return tools