import json
import os
from datetime import datetime

import openai
from dotenv import load_dotenv, find_dotenv
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from common.logger_setup import configure_logger

from requests import get

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")

# TIP: The context is provided between ``` characters. You might wonder why?
# If markdown content would be passed then ### is fragment of markdown (### is used to create a heading level 3).

system_template = """

Answer questions as truthfully as possible using the context below and nothing else 
If you don't know the answer, say: I don't know.

context: ```{context_value}``` """

user_template = """{user_question} """

function_tools = [
    {
        'type': 'function',
        'function': {
            'name': 'ToDo',
            'description': 'Add task to ToDo list. Must be apply for all task without given date',
            'parameters': {
                'type': 'object',
                'properties': {
                    'desc': {
                        'type': 'string',
                        'desc': 'Brief task description',
                    }
                },
                'required': ['desc']
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'Calendar',
            'description': 'Add event to calendar. Must be apply for all event and task with given date',
            'parameters': {
                'type': 'object',
                'properties': {
                    'desc': {
                        'type': 'string',
                        'description': 'Description of event with date',
                    },
                    'date': {
                        'type': 'string',
                        'description': 'Date of task as string in format YYYY-MM-DD e.g. 2023-11-20',
                    }
                },
                'required': ['desc', 'date']
            },
        }
    }
]


def add_todo(tool, desc):
    response = {
        'tool': tool,
        'desc': desc
    }
    return response


def add_task(tool, desc, date):
    response = {
        'tool': tool,
        'desc': desc,
        'date': date,
    }
    return response


def run_function_call(tools, user_question, system_message):
    function_definition = tools
    user = {"role": "user", "content": user_question}
    system = {"role": "system", "content": system_message}
    messages = [user, system]

    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-1106",
        model="gpt-4",
        messages=messages,
        tools=tools,
        # tool_choice="auto",  # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message
    ic(response_message)
    function_call = response_message["tool_calls"]
    ic(len(function_call))

    # if function_call response
    if function_call:
        # available functions to choose
        available_functions = {
            "ToDo": add_todo,
            "Calendar": add_task,
        }

        # get the name of the function to call
        function = function_call[0]["function"]
        function_name = function["name"]
        # get arguments of the function

        function_args = function["arguments"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(function_call[0]["function"]["arguments"])

        all_arguments = {}
        function_name = {'tool': function_name}
        all_arguments = function_name | function_args

        ic(function_args)
        function_response = function_to_call(**all_arguments)

        return function_response


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    log = configure_logger("tools")

    current_date_time = datetime.now()
    formatted_date = current_date_time.strftime('%Y-%m-%d')

    system_context = f"Context:```\n Today is {formatted_date}\n```"

    try:
        question = 'Jutro mam spotkanie z Marianem'
        answer = run_function_call(function_tools, question, system_context)
        ic(answer)
        question = 'Pojutrze mam kupić 1kg ziemniaków'
        answer = run_function_call(function_tools, question, system_context)
        ic(answer)

    except Exception as e:
        log.exception(f'Exception: {e}')
