import json
from dotenv import load_dotenv
from openai import OpenAI
from modules.Calendly import CalendlyAPI

load_dotenv()

client = OpenAI()

calendly = CalendlyAPI()

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_user_schedule_availability",
                "description": "Returns a the assistants schedule availability when asked what times they are available",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_scheduling_link",
                "description": "Creates a secheduling link that allows a user to set up a meeting with the assistant",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dummy_function",
                "description": "A dummy function that does nothing",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [],
                },
            },
        },
    ]

def dummy_function():
    return "dummy function"

def get_available_functions():
    return {
            "get_user_schedule_availability": calendly.get_user_schedule_availability,
            "create_scheduling_link": calendly.create_scheduling_link,
            "dummy_function": dummy_function,
        } 

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "Can we set up a meeting?"}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=get_tools(),
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = get_available_functions() 
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_response = function_to_call()
            print(function_response)

run_conversation()
    