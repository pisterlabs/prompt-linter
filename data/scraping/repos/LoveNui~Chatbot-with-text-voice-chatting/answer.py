import openai

from langchain.agents import AgentType, initialize_agent
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI, SerpAPIWrapper
from dotenv import load_dotenv

from bot_src.private_env import OPENAI_KEY, SERP_API_KEY

import threading
import json
import requests
import os

load_dotenv()

openai.api_key = OPENAI_KEY
os.environ["SERPER_API_KEY"] = SERP_API_KEY

llm = OpenAI(openai_api_key= OPENAI_KEY, temperature=0)
search = SerpAPIWrapper()

ai_bot_list = ["September 2021","access to real-time","AI chatbot","I'm not connected to the Internet"]
default_answer = "I'm sorry. Unfortunately, I'm unable to provide accurate information as my internet connection is currently not stable. I will investigate further and get back to you ASAP."

def search_internet(query):
    query_data = set_answer_box(query)
    return query_data

tools = [
    Tool(
        name="Intermediate Answer",
        # func=search.run,
        func=search_internet,
        description="useful for when you need to ask with search",
    )
]
tools_organic = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent_organic = initialize_agent(tools_organic, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def merge_data(data):
    result = {}
    for key in data.keys():
        if len(str(data[key])) < 250:
            result[key] = data[key]
    return result

def set_answer_box(query):
    query_data = {}
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERP_API_KEY,
        "answer_boxes": 1
    }

    # Send the request to the SerpAPI
    response = requests.get("https://serpapi.com/search", params=params)

    # Parse the JSON response
    data = json.loads(response.text)
    if "answer_box" in data.keys():
        query_data = merge_data(data["answer_box"])
    else:
        pass
    return query_data

def langchain_func(text):
    query_data = set_answer_box(text)
    if query_data == {}:
        result_answer = agent_organic.run(text)
    else:
        result_answer = agent.run(text)
    return result_answer


def get_result_openai(message_box):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = message_box
    )
    openai_answer = response.choices[0]["message"]["content"]
    return openai_answer


def check_answer_ai_bot(sentence, word_list):
    for word in word_list:
        if word in sentence:
            return True
    return False

# main function
def geneartor_answer(message, system_prompt, text):
    message_box = message
    openai_answer = ""
    result_answer = ""
    print("---------------------- openai_answer ------------------------")
    openai_answer = get_result_openai(message_box=message_box)
    print(openai_answer)
    if "Cococa-" in openai_answer or "cococa-" in openai_answer:
        print("---------------------- Serpai_answer ------------------------")
        result_answer = langchain_func(text)
        print(result_answer)
        message_box.pop(-2)
        message_box.append({"role": "assistant", "content": result_answer})
        message_box.append({"role": "system", "content": system_prompt})
        return result_answer, message_box
    elif check_answer_ai_bot(openai_answer, ai_bot_list):
        message_box.pop(-2)
        message_box.append({"role": "assistant", "content": default_answer})
        message_box.append({"role": "system", "content": system_prompt})
        return default_answer, message_box
    else:
        message_box.pop(-2)
        message_box.append({"role": "assistant", "content": openai_answer})
        message_box.append({"role": "system", "content": system_prompt})
        return openai_answer, message_box