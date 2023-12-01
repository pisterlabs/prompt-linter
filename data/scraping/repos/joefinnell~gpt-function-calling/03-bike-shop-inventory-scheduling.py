import json
import os
from modules.Calendly import CalendlyAPI
from modules.project3.Inventory import Inventory
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
calendly = CalendlyAPI()
inventory = Inventory()

inventory.add_bike("Kona", "Model X", 5)
inventory.add_bike("Specialized", "Model Y", 3)
inventory.add_bike("Specialized", "Model Z", 3)
inventory.add_bike("Specialized", "Stump Jumper", 3)
inventory.add_bike("Transition", "Model Z", 2)

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
                "name": "purchase_bike",
                "description": "Purchases a bike that the user wishes to purchase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand":{
                            "type": "string",
                            "description": "The brand name of the bike"
                        },
                        "model": {
                            "type": "string",
                            "description": "The model of the bike"
                        },
                    },
                    "required": ["brand", "model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_quantity",
                "description": "Gets the available quantity for a bike brand and model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand":{
                            "type": "string",
                            "description": "The brand name of the bike"
                        },
                        "model": {
                            "type": "string",
                            "description": "The model of the bike"
                        },
                    },
                    "required": ["brand", "model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_models",
                "description": "Gets the available models for a bike brand.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brand":{
                            "type": "string",
                            "description": "The brand name of the bike"
                        },
                    },
                    "required": ["brand"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_available_brands",
                "description": "Gets all the brands made that are available at the store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                    "required": [],
                },
            },
        },
    ]

def get_available_functions():
    return {
            "get_user_schedule_availability": calendly.get_user_schedule_availability,
            "create_scheduling_link": calendly.create_scheduling_link,
            "purchase_bike": inventory.purchase_bike,
            "get_available_quantity": inventory.get_available_quantity,
            "get_available_models": inventory.get_available_models,
            "get_available_brands": inventory.get_available_brands,
        } 

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [
        {"role": "system", "content": "You are a bike shop owner who is working with a customer to purchase a bike."},
        {"role": "user", "content": "What models do you have for Specialized?"}
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=get_tools(),
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = get_available_functions() 
        messages.append(response_message)  # extend conversation with assistant's reply

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            arguments = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**arguments)
            print(function_response)

run_conversation()
    