import json
import os

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
            'name': 'get_population',
            'description': 'Get population of given country',
            'parameters': {
                'type': 'object',
                'properties': {
                    'parameter': {
                        'type': 'string',
                        'description': 'Name of country, e.g. poland, france, germany, usa',
                    }
                },
                'required': ['parameter']
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_exchange_price',
            'description': 'Get exchange price for given currency',
            'parameters': {
                'type': 'object',
                'properties': {
                    'parameter': {
                        'type': 'string',
                        'description': 'Code of currency, e.g. usd, eur, jpy',
                    }
                },
                'required': ['parameter']
            },
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'answer_question',
            'description': 'Answer general question witch are not connected with population or currency',
            'parameters': {
                'type': 'object',
                'properties': {
                    'parameter': {
                        'type': 'string',
                        'description': 'Repeat given question',
                    }
                },
                'required': ['parameter']
            },
        }
    }
]


def give_me_answer_based_on_context(usr_template=None,
                                    usr_question=None,
                                    sys_template=None,
                                    context_val=None,
                                    log=None):
    if log is None:
        log = configure_logger("knowledge")

    log.info(f"usr_question:{usr_question}")
    try:

        chat = ChatOpenAI(model_name="gpt-3.5-turbo")
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_template),
                ("human", usr_template),
            ]
        )
        get_name_formatted_chat_prompt = chat_prompt.format_messages(context_value=context_val,
                                                                     user_question=usr_question)
        log.info(f"prompt: {get_name_formatted_chat_prompt}")
        ai_response = chat.predict_messages(get_name_formatted_chat_prompt)
        log.info(f"content: {ai_response}")
        response_content = ai_response.content
        log.info(f"response_content: {response_content}")

        return response_content
    except Exception as e:
        log.error(f"Exception: {e}")


def get_population(country: str) -> int:
    country_data = get(f'https://restcountries.com/v3.1/name/{country}').json()[0]
    return country_data['population']


def get_exchange_price(currency: str) -> float:
    currency_data = get(f'https://api.nbp.pl/api/exchangerates/rates/c/{currency}').json()
    return currency_data['rates'][0]['ask']


def answer_question(question: str) -> str:
    return give_me_answer_based_on_context(user_template, question, system_template, "", log)


def run_function_call(tools, user_question):
    function_definition = tools
    messages = [{"role": "user", "content": user_question}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
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
            "get_population": get_population,
            "get_exchange_price": get_exchange_price,
            "answer_question": answer_question,
        }

        # get the name of the function to call
        function_name = function_call[0]["function"]["name"]
        function_to_call = available_functions[function_name]
        # get arguments of the function
        function_args = json.loads(function_call[0]["function"]["arguments"])
        ic(function_args)
        function_response = function_to_call(function_args.get("parameter"))

        return function_response


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    context = ""

    try:
        # question = 'ile orientacyjnie ludzi mieszka w Polsce?'
        # answer = run_function_call(function_tools, question)
        # ic(answer)
        log = configure_logger("knowledge")
        question = 'jak nazywa się\xa0stolica Czech'
        answer = run_function_call(function_tools, question)
        ic(answer)

    except Exception as e:
        log.exception(f'Exception: {e}')
