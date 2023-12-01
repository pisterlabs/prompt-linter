import openai
from decouple import config

from apis.random_word import get_random_word
from utils.printer import ColorPrinter as Printer

openai.api_key = config("CHATGPT_API_KEY")

JOKE_SETUP = """
You will be given a subject by the user. You will return a joke, but it should not be too long (4 lines at most). You will not provide an introduction like 'Here's a joke for you' but get straight into the joke.
There is a function called 'get_random_word'. If the user does not provide a subject, you should call this function and use the result as the subject. If the user does provide a subject, you should not call this function. The only exception is if the user asks for a random joke, in which case you should call the function and use the result as the subject.
Example: {user: 'penguins'} = Do not call the function => provide a joke about penguins.
Example: {user: ''} = Call the function => provide a joke about the result of the function.
Example: {user: 'soul music'} = Do not call the function => provide a joke about soul music.
Example: {user: 'random'} = Call the function => provide a joke about the result of the function.
Example: {user: 'guitars'} = Do not call the function => provide a joke about guitars.
Example: {user: 'give me a random joke'} = Call the function => provide a joke about the result of the function.
"""


def get_joke_result(query):
    messages = [
        {"role": "system", "content": JOKE_SETUP},
        {"role": "user", "content": query},
    ]

    functions = [
        {
            "name": "get_random_word",
            "description": "Get a subject for your joke.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number_of_words": {
                        "type": "integer",
                        "description": "The number of words to generate.",
                    }
                },
            },
        }
    ]

    first_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default
    )["choices"][0]["message"]
    messages.append(first_response)

    if first_response.get("function_call"):
        function_response = get_random_word()

        messages.append(
            {
                "role": "function",
                "name": "get_random_word",
                "content": function_response,
            }
        )

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )["choices"][0]["message"]
        messages.append(second_response)
        Printer.color_print(messages)
        return second_response["content"]

    Printer.color_print(messages)
    return first_response["content"]


print(get_joke_result(""))
