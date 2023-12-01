# encoding: UTF-8
import os
import re
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_base = os.getenv("API_BASE")
openai.api_key = os.getenv("API_KEY")
model = 'gpt-3.5-turbo-0613'


def add_decimal_values(arguments):
    value1 = int(re.search(r'"value1": (\d+)', str(arguments)).group(1))
    value2 = int(re.search(r'"value2": (\d+)', str(arguments)).group(1))

    result = value1 + value2
    print(f"{value1} + {value2} = {result} (decimal)")

    return value1 + value2


def add_hexadecimal_values(arguments):
    value1 = re.search(r'"value1": "(\w+)"', str(arguments)).group(1)
    value2 = re.search(r'"value2": "(\w+)"', str(arguments)).group(1)

    decimal1 = int(value1, 16)
    decimal2 = int(value2, 16)

    result = hex(decimal1 + decimal2)[2:]
    print(f"{value1} + {value2} = {result} (hex)")
    return result

def get_completion(messages):
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        functions = [
            {
                "name": "add_decimal_values",
                "description": "Add two decimal values",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value1": {
                            "type": "integer",
                            "description": "The first decimal value to add. For example, 5",
                        },
                        "value2": {
                            "type": "integer",
                            "description": "The second decimal value to add. For example, 10",
                        },
                    },
                    "required": ["value1", "value2"],
                },
            },
            {
                "name": "add_hexadecimal_values",
                "description": "Add two hexadecimal values",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value1": {
                            "type": "string",
                            "description": "The first hexadecimal value to add. For example, 5",
                        },
                        "value2": {
                            "type": "string",
                            "description": "The second hexadecimal value to add. For example, A",
                        },
                    },
                    "required": ["value1", "value2"],
                },
            },
        ],
        temperature = 0,
    )
    return response


QUESTION = (
    "What's the result of 22 plus 5 in decimal added to the hexadecimal number A?"
)
messages = [
    {"role": "user", "content": QUESTION},
]

while True:
    response = get_completion(messages)

    if response.choices[0]["finish_reason"] == "stop":
        print(response.choices[0]["message"]["content"])
        break

    elif response.choices[0]["finish_reason"] == "function_call":
        fn_name = response.choices[0].message["function_call"].name
        arguments = response.choices[0].message["function_call"].arguments

        function = locals()[fn_name]
        result = function(arguments)

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": fn_name,
                    "arguments": arguments,
                },
            }
        )

        messages.append(
            {
                "role": "function",
                "name": fn_name,
                "content": f'{{"result": {str(result)} }}'}
        )

        response = get_completion(messages)
