import openai
import os
import json
from datetime import datetime
import pytz

os.environ["OPENAI_API_KEY"] = "opena api key"
local_time = datetime.now()
def calculate_time_difference(local_time, location):
    local_time = datetime.strptime(local_time, '%Y-%m-%dT%H:%M:%S')

    
    location_time = datetime.now(pytz.timezone(location))

    
    local_time_only = local_time.time()
    location_time_only = location_time.time()

    
    time_difference_in_minutes = ((location_time_only.hour * 60 + location_time_only.minute) -
                                  (local_time_only.hour * 60 + local_time_only.minute))

    hours, minutes = divmod(abs(time_difference_in_minutes), 60)

    
    if time_difference_in_minutes < 0:
        time_difference_str = f'{hours} hours, {minutes} minutes behind local time'
    else:
        time_difference_str = f'{hours} hours, {minutes} minutes ahead of local time'

    
    location_time_str = location_time.strftime('%H:%M:%S')

    time_response={
        "local_time":  location_time_str,
        "location_time": time_difference_str
    }

    return json.dumps(time_response)

local_time_str = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')  
time_response_json = calculate_time_difference(local_time_str, 'Australia/Brisbane')  
time_response = json.loads(time_response_json)

print(f"Time at location: {time_response['local_time']}")
print(f"Time difference: {time_response['location_time']}")
def conversation_with_time():
    location = input("Please enter your location: ")
   
    messages = [{"role": "user", "content": f"What is the time of day in {location}?"}]
    functions = [
        {
            "name": "calculate_time_difference",
            "description": "Calculates the time difference for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "local_time": {
                      "type": "string",
                      "description": "Current local time of the user request"
                    },
                    "location": {
                        "type": "string",
                        "description": "The city, state or country, needs to be the timezone like US/Pacific",
                    },
                },
                "required": ["local_time","location"],
            },
        }
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  
    )
    response_message = response.choices[0].message

    
    if response_message.function_call:
        
        available_functions = {
            "calculate_time_difference": calculate_time_difference,
        }  
        function_name = response_message.function_call.name
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message.function_call.arguments)
        function_response = fuction_to_call(
            local_time=function_args.get("local_time"),
            location=function_args.get("location")
        )

        
        messages.append(response_message)  
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  
        second_response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  
        return second_response


print(conversation_with_time())
