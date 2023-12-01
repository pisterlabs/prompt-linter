import os
from langchain.agents import Tool
from dotenv import load_dotenv
from langchain.utilities import BingSearchAPIWrapper
import aiohttp
import requests

search = BingSearchAPIWrapper()

def search_bing(query):
    subscription_key = str(os.getenv("BING_SUBSCRIPTION_KEY"))
    endpoint = str(os.getenv("BING_SEARCH_URL"))
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "count": 3}
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    json_response = response.json()
    return str(json_response)[:1000]

async def async_search_bing(query):
    subscription_key = str(os.getenv("BING_SUBSCRIPTION_KEY"))
    endpoint = str(os.getenv("BING_SEARCH_URL"))
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": query, "count": 3}
    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, headers=headers, params=params) as response:
            response.raise_for_status()
            json_response = await response.json()
            return str(json_response)[:1000]

def BingTool():
    tools = []
    tools.append(Tool(
        name = "search",
        func=search_bing,
        description="useful for when you need to answer questions about current events",
        coroutine=async_search_bing
    ))
    return tools

def bing_search():
    tools = []
    tools.extend(BingTool())
    return tools