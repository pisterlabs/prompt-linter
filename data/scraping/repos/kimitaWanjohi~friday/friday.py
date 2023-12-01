import os
import json
import openai
import environs
import threading
import webbrowser
from .speech import say
from .listen import listen
from .typing import typing

env = environs.Env()

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env.read_env(os.path.join(parent_dir, ".env"))

openai.api_key = env("OPENAI_API_KEY")

history = [
    {"role": "system", "content": "My name is Friday. I am an AI created by Iron man."},
    {"role": "system", "content": "I am here to help you with your daily tasks."},
]


def open_website(url):
    webbrowser.open(url)


functions = [
    {
        "name": "open_website",
        "description": "Opens a website in the browser",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the website to open",
                },
            },
            "required": ["url"],
        },
    }
]

def completion_gpt3():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=history,
        functions=functions,
        function_call="auto"
    )
    return completion.choices[0]["message"]

def completion_gpt4():
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=history
    )
    return completion.choices[0]["message"]


def run_threaded(content):
    thread1 = threading.Thread(target=typing, args=(content,))
    thread2 = threading.Thread(target=say, args=(content,))
    thread1.start()
    thread2.start()
    
def chat():
    say("Hello, my name is Friday. How can I help you today?")
    while True:

        audio = listen()
        if audio == "exit":
            break
        history.append({
            "role": "user",
            "content": audio
        })
        gpt_3 = completion_gpt3()        
        
        if gpt_3.get("function_call"):
            available_functions = {
                "open_website": open_website,
            }
            function_name = gpt_3["function_call"]["name"]
            fuction_to_call = available_functions[function_name]
            function_args = json.loads(gpt_3["function_call"]["arguments"])
            function_response = fuction_to_call(
                url=function_args.get("url"),
            )
            history.append(gpt_3)
            history.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  
        print()
        if gpt_3.content == "None":
            say("Done! Can I help you with anything else?")
            continue
        else:
            say(gpt_3.content)
            # print("Friday: {}".format(gpt_3.content), end="", flush=True)

chat()