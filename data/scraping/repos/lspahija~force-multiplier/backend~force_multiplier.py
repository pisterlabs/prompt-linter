import logging
import os
from types import SimpleNamespace
import openai
import json

from mock import get_mock_completion
from prompts import get_system_prompt, get_openai_functions, get_user_prompt

MOCK_COMPLETION = os.getenv("MOCK_COMPLETION", False)
USE_OPENAI_FUNCTIONS = os.getenv("USE_OPENAI_FUNCTIONS", False)


async def apply_feedback(document, document_is_code, feedback, api_key):
    diff = await get_diff(document=document, document_is_code=document_is_code, feedback=feedback, api_key=api_key)
    modified_document = apply_diff(document=document, diff=diff)
    return modified_document


async def get_diff(document, document_is_code, feedback, api_key):
    messages = [
        {
            "role": "system",
            "content": get_system_prompt(document_is_code)
        },
        {
            "role": "user",
            "content": get_user_prompt(document_is_code, document, feedback)
        }
    ]

    completion = get_mock_completion(document) if MOCK_COMPLETION else await get_completion(messages, api_key)
    logging.info(completion)

    return json.loads(remove_newlines(completion), object_hook=lambda d: SimpleNamespace(**d))


def remove_newlines(string):
    return string.replace('\n', '')


async def get_completion(messages, api_key):
    arguments = {
        "model": "gpt-4-0613",
        "messages": messages,
        "api_key": api_key,
        "timeout": 15,
        "temperature": 0.2,
    }

    if USE_OPENAI_FUNCTIONS:
        arguments["functions"] = get_openai_functions()

    res = await openai.ChatCompletion.acreate(**arguments)
    first_choice = res.choices[0]

    try:
        return process_result(first_choice)
    except InadequateFeedbackException as e:
        print(str(e))
        raise e


def process_result(res):
    if not USE_OPENAI_FUNCTIONS:
        return res.message.content

    if res.finish_reason != "function_call":
        raise InadequateFeedbackException(res.message.content)

    function_call = res.message.function_call

    if function_call.name != "apply_diff":
        raise InadequateFeedbackException(function_call.arguments)

    return function_call.arguments


def apply_diff(document, diff):
    for change in diff.diff:
        start = change.start
        end = change.end
        replacement = change.replacement

        block_start_index = document.find(start)
        if block_start_index != -1:
            remaining_document = document[block_start_index + len(start):]
            block_end_index = remaining_document.find(end) + len(end)

            if block_end_index != -1:
                document = document[:block_start_index] + replacement + remaining_document[block_end_index:]

    return document


class InadequateFeedbackException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
