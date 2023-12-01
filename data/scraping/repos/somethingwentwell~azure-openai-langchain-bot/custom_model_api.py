import os
from langchain.agents import Tool
from dotenv import load_dotenv
import aiohttp
import requests
import json

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
load_dotenv(os.path.join(BASEDIR, '.env'), override=True)

def custom_api_call(question):
        try:
            url = f"{str(os.getenv('CUSTOM_LLM_API_URL'))}"
            payload = json.dumps({   
                "question": f"User: {question} \nAI: "  
            }  )

            response = requests.post(url,  data=payload)
            response_json = response.json()

            return response_json
                    
        except Exception as e:
                print(f"Error: {e}")
                return f"Error: {e}"


def custom_api():
    tools = []
    tools.append(Tool(
        name = "Direct call Custom LLM Model",
        func=custom_api_call,
        description="Useful for when you need to answer questions using Custom LLM Model. Input must be the exact same text as user's ask.",
    ))
    return tools

def custom_model_api():
    tools = []
    tools.extend(custom_api())
    return tools