from pprint import pprint
from typing import Dict, List
from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
import openai
from langchain.chat_models import AzureChatOpenAI
import re  # regular expressions to parse the data
from typing import Dict
import inspect
import requests
import json
from memory import (
    get_chat_history,
    load_conversation_history,
    log_bot_message,
    log_user_message,
)
import pytz
from datetime import datetime
import math
import python_weather



json_file_path = "chat_histories/megazonecloud1.json"

load_dotenv()

# Retrieve the environment variables using os.getenv
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
# OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")
# OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
# OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
# OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
# deployment_id = OPENAI_DEPLOYMENT_NAME
# deployment_name = OPENAI_DEPLOYMENT_NAME

app = FastAPI()

import requests


def calculator(num1, num2, operator):
    if operator == '+':
        return str(num1 + num2)
    elif operator == '-':
        return str(num1 - num2)
    elif operator == '*':
        return str(num1 * num2)
    elif operator == '/':
        return str(num1 / num2)
    elif operator == '**':
        return str(num1 ** num2)
    elif operator == 'sqrt':
        return str(math.sqrt(num1))
    else:
        return "Invalid operator"


def get_current_time(location):
    try:
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        # Get the current date in the timezone
        current_date = now.strftime("%Y-%m-%d")

        return f"Current time: {current_time}, Current date: {current_date}"
    except:
        return "Sorry, I couldn't find the timezone for that location."

# Function to delete the data from the JSON file.
# def delete_json_data():
#     with open(json_file_path, "w") as f:
#         f.write("")

# Function to handle the refresh trigger.
# def handle_refresh_trigger():
    
#     return True

# Check if the refresh trigger is activated and delete the data 
# if handle_refresh_trigger():
#     delete_json_data()


    
def create_booking(booking_subject, room_id, applicant_name, date, start_time, end_time, duration, attendees):
    url = "http://4.230.139.78:8080/create_booking"
    data = {
        "booking_subject": booking_subject,
        "room_id": room_id,
        "applicant_name": applicant_name,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "attendees": attendees
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return "Booking created successfully"
    else:
        return "Failed to create booking"
    

def read_booking(booking_id):
    url = f"http://4.230.139.78:8080/read_booking/{booking_id}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read booking"
    
def find_booking_by_time(date, start_time):
    url = "http://4.230.139.78:8080/read_booking_by_date_and_time"
    params = {"date": date, "start_time": start_time}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by date and time"

def find_booking_by_date(date):
    url = "http://4.230.139.78:8080/read_booking_by_date"
    params = {"date": date}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by date"

def find_booking_by_applicant_name(applicant_name):
    url = "http://4.230.139.78:8080/read_booking_by_applicant_name"
    params = {"applicant_name": applicant_name}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read booking by applicant name"

    
def read_all_bookings():
    url = "http://4.230.139.78:8080/read_all_bookings"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read all bookings"
    
def available_rooms(date, start_time, end_time, attendees):
    url = "http://4.230.139.78:8080/available_rooms"
    params = {"date": date, "start_time": start_time, "end_time": end_time, "attendees": attendees}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to read available rooms"
    
def get_all_meeting_rooms():
    url = "http://4.230.139.78:8080/all_meeting_rooms"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to read all meeting rooms"
    
def update_booking(booking_subject, date, start_time, booking_request):
    url = f"http://4.230.139.78:8080/update_booking"
    data = {
        "booking_subject": booking_subject,
        "date": date,
        "booking_request": {
            "booking_subject": booking_request.booking_subject,
            "room_id": booking_request.room_id,
            "applicant_name": booking_request.applicant_name,
            "date": booking_request.date,
            "start_time": booking_request.start_time,
            "end_time": booking_request.end_time,
            "duration": booking_request.duration,
            "attendees": booking_request.attendees
        }
    }
    response = requests.put(url, json=data)

    if response.status_code == 200:
        return "Booking updated successfully"
    else:
        return "Failed to update booking"


def get_coffee_menu():
    url = "http://4.230.139.78:8080/coffee_menu"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to get coffee menu"

def get_coffee_orders():
    url = "http://4.230.139.78:8080/coffee_orders"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return "Failed to get coffee orders"
    

def order_coffee(customer_name, coffee_type, num_coffees, order_date, order_time, expected_time, pickup_status):
    url = "http://4.230.139.78:8080/order_coffee"
    data = {
        "customer_name": customer_name,
        "coffee_type": coffee_type,
        "num_coffees": num_coffees,
        "order_date": order_date,
        "order_time": order_time,
        "expected_time": expected_time,
        "pickup_status": pickup_status
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        return "Coffee ordered successfully"
    else:
        return "Failed to order coffee"


def recommend_room_by_meeting_name(meeting_name: str):
    url = "http://4.230.139.78:8080/recommend_room_by_meeting_name"
    params = {"meeting_name": meeting_name}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json() 
    else:
        return "Failed to recommend meeting room by meeting name"

def delete_booking_based_on_title_and_time(booking_subject, date, start_time):
    url = "http://4.230.139.78:8080/delete_booking"
    params = {"booking_subject": booking_subject, "date": date, "start_time": start_time}
    response = requests.delete(url, params=params)

    if response.status_code == 200:
        return "Booking deleted successfully"

def delete_booking_based_on_date_and_time(date, start_time):
    url = "http://4.230.139.78:8080/delete_booking_with_date_and_time"
    params = {"date": date, "start_time": start_time}
    response = requests.delete(url, params=params)

    if response.status_code == 200:
        return "Booking deleted successfully"

# Load the functions frm the functions.json file
with open("functions.json", "r") as f:
    functions = json.load(f)

# Extract the names of all functions
function_names = [f['name'] for f in functions]

# Print the function names
print(function_names)


# Define the available functions

available_functions = {
    "create_booking": create_booking,
    "read_booking": read_booking,
    "read_all_bookings": read_all_bookings,
    "update_booking": update_booking,
    "get_coffee_menu": get_coffee_menu,
    "order_coffee": order_coffee,
    "get_all_meeting_rooms": get_all_meeting_rooms,
    "get_coffee_orders": get_coffee_orders,
    "get_current_time": get_current_time,
    "calculator": calculator,
    "available_rooms": available_rooms,
    "recommend_room_by_meeting_name":recommend_room_by_meeting_name,
    "find_booking_by_time": find_booking_by_time,
    "delete_booking_based_on_title_and_time": delete_booking_based_on_title_and_time,
    "delete_booking_based_on_date_and_time":delete_booking_based_on_date_and_time,
    "find_booking_by_date": find_booking_by_date,
    "find_booking_by_applicant_name":find_booking_by_applicant_name
}

# helper method used to check if the correct arguments are provided to a function
def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True

openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.api_base = "https://api.openai.com"

def run_multiturn_conversation(messages, functions, available_functions):
    # Step 1: send the conversation and available functions to GPT

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=messages,
        functions= functions,
        function_call="auto", 
        temperature=0
    )


    # Step 2: check if GPT wanted to call a function
    while response["choices"][0]["finish_reason"] == 'function_call':
        response_message = response["choices"][0]["message"]
        print("Recommended Function call:")
        print(response_message.get("function_call"))
        print()
        
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        
        function_name = response_message["function_call"]["name"]
        
        # verify function exists
        if function_name not in available_functions:
            return "Function " + function_name + " does not exist"
        function_to_call = available_functions[function_name]  
        
        # verify function has correct number of arguments
        function_args = json.loads(response_message["function_call"]["arguments"])
        if check_args(function_to_call, function_args) is False:
            return "Invalid number of arguments for function: " + function_name
        function_response = function_to_call(**function_args)
        # convert function_response to string
        function_response = str(function_response)
        
        print("Output of function call:")
        print(function_response)
        print()
        
        # Step 4: send the info on the function call and function response to GPT
        
        # adding assistant response to messages
        messages.append(
            {
                "role": response_message["role"],
                "function_call": {
                    "name": response_message["function_call"]["name"],
                    "arguments": response_message["function_call"]["arguments"],
                },
                "content": None
            }
        )

        # adding function response to messages
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        print("Messages in next request:")
        for message in messages:
            print(message)
        print()

        response = openai.ChatCompletion.create(
            model = "gpt-4-1106-preview",
            messages=messages,
            # deployment_id=deployment_name,
            function_call="auto",
            functions=functions,
            temperature=0
        )  # get a new response from GPT where it can see the function response

    return response

class ChatRequest(BaseModel):
    user_input: str


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r", encoding='utf-8') as f:
        prompt_template = f.read()

    return prompt_template

   
@app.post("/chat_response/{conversation_id}")
async def get_bot_response(req: ChatRequest, conversation_id: str) -> Dict[str, str]:
    history_file = load_conversation_history(conversation_id)
    chat_history = get_chat_history(conversation_id)
    # Construct the next prompt based on the user input and the chat history
    #next_prompt = construct_next_prompt(req.user_input, chat_history)
    messages = [
            {
                "role": "system",
                #"content": next_prompt
                "content": "Here is your chat history:\n" + chat_history
            },
            {
                "role": "system",
                "content": """ Always think that the user name is 양동준. Always think the date is today date if the user doesn't give any specific dates. You always call the function, get_current_time of Asia/Seoul to check the current date and time before booking the meeting room according to user requirements and you must always reference the previous conversation with the user to ensure a coherent and relevant response for the next interaction.  """ },
            {
                "role": "system",
                "content": """ The user has requested to book a meeting room at a time when our standard rooms
                                are fully booked. Explore alternative options such as recommending meeting rooms
                                in a different building such as megazone building or suggesting smaller rooms where extra
                                chairs can be added if the number of attendees is slightly higher than the room's capacity. When providing alternatives, explain to the user why each option is suitable. For example,
                                if recommending a room in a different building, mention the amenities and proximity. If 
                                suggesting adding chairs to a smaller room, reassure the user that comfort and space will still
                                be adequate. The assistant should only answer questions related to its assigned task. If asked about other things, the assistant should respond with 'I am sorry, because it's not my assigned task so I can't help you.' This is to prevent hallucinations and prompt injection. """ },
           
            {
                "role": "system",
                "content": "You are an AI assistant that helps with meeting room bookings and coffee orders for the user named 양동준 who is working at MegaZone Cloud company which is located in Seoul, South Korea and always gives reasons for your answers and how did you get those answers. Always use Korean language to communicate with the user.",
            },
            {
                "role": "user",
                "content": req.user_input
            }
        ]
    
    response = run_multiturn_conversation(messages, functions, available_functions)
    log_user_message(history_file, req.user_input)
    log_bot_message(history_file, str(response))

    # return {"bot_response": response["output"]}
    print("Response from GPT:")
    print(response)
    if "choices" in response and response["choices"]:
        print(response["choices"][0]["message"]["content"])
    else:
        print("No choices found in response.")
    # Step 5: return the response from GPT
    return {"bot_response": response["choices"][0]["message"]["content"]}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
