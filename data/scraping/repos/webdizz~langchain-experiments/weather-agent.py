from dotenv import load_dotenv, find_dotenv, get_key

_ = load_dotenv(find_dotenv())

from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import requests
import langchain


def get_ip():
    """Returns client public IP address"""
    try:
        response = requests.get("https://ifconfig.me")
        client_ip = response.text
    except Exception as e:
        print(f"Failed to get client IP: {e}")
        raise
    return client_ip

def get_city(text: str) -> str:
    """Returns information about city using current IP address.
    Use this for any questions related to knowing city or town.
    """
    ip = get_ip()
    ip2location_api_key = get_key(find_dotenv(), "IP2LOCATION_API_KEY")
    city_by_ip_url = f"https://api.ip2location.io/?key={ip2location_api_key}&ip={ip}"
    try:
        response = requests.get(city_by_ip_url)
        city = response.json()["city_name"]
    except Exception as e:
        print(f"Failed to get city by IP: {e}")
        raise
    return city

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search. If temperature is in Fahrenheit, please convert it to Celsius and provide the final answer.",
    ),
    Tool(
        name="City Finder",
        func=get_city,
        description="useful for when you need to know city or town",
    ),
]

self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
)

# langchain.debug = True
self_ask_with_search.run("What is a weather today?")
