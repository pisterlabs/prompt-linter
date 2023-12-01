import json
import os
from typing import Any, Callable

import dotenv
import openai

if not dotenv.load_dotenv(".env"):
    raise ValueError("EnvironmentError: Failed to load `.env`")

api_key = os.getenv("OPENAI_API_KEY") or ""

if not api_key:
    raise ValueError("EnvironmentError: Failed to load `OPENAI_API_KEY`")

openai.api_key = api_key


def get_current_weather(location: str, unit: str = "celsius"):
    # This is a mock function, so we return a mock weather report.
    weather_report = f"The current weather in {location} is 20 degrees {unit}."
    return weather_report


function_factory = {
    "get_current_weather": get_current_weather,
}

function_definitions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


def get_function(
    message: dict[str, str], functions: dict[str, Callable]
) -> dict[str, Any]:
    # Note: the JSON response may not always be valid; be sure to handle errors
    name = message["function_call"]["name"]
    callback = functions[name]
    arguments = json.loads(message["function_call"]["arguments"])
    content = callback(**arguments)
    response = {
        "role": "function",
        "name": name,
        "content": content,
    }
    return response


def initialize_session() -> list[dict[str, str]]:
    system_prompt = {
        "role": "system",
        "content": "My name is ChatGPT. I am a helpful assistant.",
    }
    user_input = {
        "role": "user",
        "content": "Hello! What is your name?",
    }
    messages = [system_prompt, user_input]
    return messages


messages = initialize_session()


def add_message(role: str, content: str) -> dict[str, str]:
    message = {"role": role, "content": content}
    messages.append(message)
    return message


def add_completion(
    response: dict[str, Any], messages: list[dict[str, str]]
) -> dict[str, str]:
    message = response["choices"][0]["message"]
    messages.append(message)
    return message


def print_completion(messages: list[dict[str, str]]) -> None:
    for message in messages:
        print("keys:", message.keys())
        print("role:", message["role"])
        if message["role"] in ["system", "user", "assistant"]:
            print("content:", message["content"])
            function_call = message.get("function_call", None)
            if function_call:
                print("function_call:", function_call)
        else:
            print("name:", message["name"])
            print("content:", json.dumps(message["content"]))
        print()


get_chat_completion = openai.ChatCompletion.create

if __name__ == "__main__":
    response = get_chat_completion(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    message = add_completion(response, messages)
    message = add_message(
        "user", "What is the weather like today in New York City, New York?"
    )

    response = get_chat_completion(  # This line calls the function
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_definitions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    print(response)
    message = add_completion(response, messages)  # This line appends the function call
    function = get_function(
        message, function_factory
    )  # This line gets the function result
    messages.append(function)  # This line appends the function result

    response = get_chat_completion(  # This line prompts GPT with the function results
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_definitions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    print(response)  # The model responds as expected
    add_completion(response, messages)  # We append the models response
    add_message("user", "Thank you! How should I dress for that kind of temperature?")

    response = get_chat_completion(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=function_definitions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    add_completion(response, messages)

    print_completion(messages)
