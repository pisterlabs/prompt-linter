import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# example dummy function acting as weather API
def get_current_weather(location, unit='fahrenheit'):
    """ Get the weather in a given location """
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

def run_conversation():
    messages = [{"role": "user", "content": "Whats the weather like in boston?"}]
    
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the weather in a given location",
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

    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0613",
                        messages=messages,
                        functions=functions,
                        function_call="auto",
    )

    response_message = response.choices[0].message

    if response_message.get("function_call"):
        available_functions = {
            "get_current_weather": get_current_weather,
        }  
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )

        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response
    
print(run_conversation())