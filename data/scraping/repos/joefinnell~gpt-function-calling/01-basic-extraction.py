import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def format_vehicle_info(year=None, make=None, model=None, color=None, trim=None):
    vehicle_info = {
        "year": year,
        "make": make,
        "model": model,
        "color": color,
        "trim": trim
    }
    return json.dumps(vehicle_info)


def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "Do you have any information about a 2023 blue subaru outback trail editions or a toyota tacoma?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "format_vehicle_info",
                "description": "Returns a formatted string containing the year, make, model, color, and trim of a vehicle",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer"},
                        "make": {"type": "string"},
                        "model": {"type": "string"},
                        "color": {"type": "string"},
                        "trim": {"type": "string"},
                    },
                    "required": ["make"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "format_vehicle_info": format_vehicle_info,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                year=function_args.get("year"),
                make=function_args.get("make"),
                model=function_args.get("model"),
                color=function_args.get("color"),
                trim=function_args.get("trim"),
            )
            print(function_response)
print(run_conversation())
    