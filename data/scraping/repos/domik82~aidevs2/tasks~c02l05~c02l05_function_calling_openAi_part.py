import os

import openai
import json

from dotenv import load_dotenv, find_dotenv
from icecream import ic

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

function_definition_add_user = {
    "name": "addUser",
    "description": "addUser",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of user"
            },
            "surname": {
                "type": "string",
                "description": "Name of user"
            },
            "year": {
                "type": "integer",
                "description": "Name of user"
            }
        }
    }
}


def add_user(name, surname, year):
    ic(f"Executed with params: {name}, {surname}, {year}")
    return 'added user'


def run_function_add_name():
    function_definition = [function_definition_add_user]
    messages = [{"role": "user", "content": "Dodaj uzytkownika Jan Kowalski rok urodzenia 1982"}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        functions=function_definition,
        # tool_choice="auto",  # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message
    ic(response_message)
    function_call = response_message["function_call"]
    # if function_call response
    if function_call:
        # available functions to choose
        available_functions = {
            "addUser": add_user,
        }

        # get the name of the function to call

        function_name = function_call["name"]
        function_to_call = available_functions[function_name]
        # get arguments of the function
        function_args = json.loads(response_message["function_call"]["arguments"])
        ic(function_args)
        function_response = function_to_call(
            name=function_args.get("name"),
            surname=function_args.get("surname"),
            year=function_args.get("year"),
        )
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response


ic(run_function_add_name())
