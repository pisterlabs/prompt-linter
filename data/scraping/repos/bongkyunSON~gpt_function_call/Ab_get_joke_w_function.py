import openai
import json
import os
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from apis.random_word import get_random_word
from utils.printer import ColorPrinter as Printer

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

JOKE_SETUP = """
사용자에 의해 주제가 주어집니다. 농담을 돌려주겠지만 너무 길지 않아야 합니다(최대 4줄). '여기 농담이 있습니다'와 같은 소개를 하지 않고 바로 농담으로 들어갑니다.
get_random_word'라는 함수가 있습니다. 사용자가 주어를 제공하지 않으면 이 함수를 호출하고 결과를 주어로 사용해야 합니다. 만약 사용자가 주어를 제공한다면 이 함수를 호출하면 안 됩니다. 유일한 예외는 사용자가 임의의 농담을 요청하는 경우이며, 이 경우 함수를 호출하고 결과를 주어로 사용해야 합니다.
예: {user: 'penguins'} = 함수를 호출하지 마십시오 => 펭귄에 대한 농담을 제공합니다.
예: {user: '''} = 함수 호출하기 => 함수 결과에 대한 농담을 제공합니다.
예: {user: 'soul 음악'} = 기능을 호출하지 마십시오 => 소울 음악에 대한 농담을 제공합니다.
예: {user: 'random'} = 함수 호출 => 함수 결과에 대한 농담을 제공합니다.
예: {user: 'guitars'} = 기능을 호출하지 마십시오 => 기타에 대한 농담을 제공합니다.
예: {user: '막말 장난 주세요'} = 함수 호출하기 => 함수 결과에 대한 농담을 제공합니다.
"""


def get_joke_result(query):
    messages = [
        {"role": "system", "content": JOKE_SETUP},
        {"role": "user", "content": query},
    ]

    functions = [
        {
            "name": "get_random_word",
            "description": "농담에 대한 주제를 잡으세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number_of_words": {
                        "type": "integer",
                        "description": "생성할 단어 수 입니다.",
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


try:
    while True:
        print(
            "원하는 농담 주제를 입력하세요"
        )
        user_quote = input("엔터를 눌러 입력하거나 끝내시려면 ctrl + c를 입력하세요: ")
        result = get_joke_result(user_quote)
except KeyboardInterrupt:
    print("종료중...")





