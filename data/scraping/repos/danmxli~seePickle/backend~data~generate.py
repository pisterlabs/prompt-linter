from data.mockdata import mockdata
import os
import cohere
from dotenv import load_dotenv
import time
from data.filter import filter_text, filter_documents
load_dotenv('.env')

# True to call cohere api, False to send mockdata
call_api = True

cohere_key = os.getenv('COHERE_KEY')
if cohere_key is not None and cohere_key != '':
    co = cohere.Client(cohere_key)
else:
    ...


def base_plan(userPrompt, prompt_type):

    base_details = {}

    if call_api:
        ...
        if prompt_type == 'prompt_quickstart':
            response = prompt_quickstart(userPrompt)
            raw_text = response.text  # type: ignore
            base_details["raw"] = raw_text
            base_details["task_list"] = filter_text(raw_text)
            base_details["resource_list"] = []

        elif prompt_type == 'prompt_developer':
            response = prompt_developer(userPrompt)
            raw_text = response.text  # type: ignore
            raw_resources = response.documents  # type: ignore
            base_details["raw"] = raw_text
            base_details["task_list"] = filter_text(raw_text)
            base_details["resource_list"] = filter_documents(raw_resources)

        elif prompt_type == 'prompt_academia':
            response = prompt_academia(userPrompt)
            raw_text = response.text  # type: ignore
            raw_resources = response.documents  # type: ignore
            base_details["raw"] = raw_text
            base_details["task_list"] = filter_text(raw_text)
            base_details["resource_list"] = filter_documents(raw_resources)

        return base_details

    else:
        ...
        # receive mockdata
        mock = mockdata()
        time.sleep(3)

        base_details["raw"] = mock["raw_text"]
        base_details["task_list"] = filter_text(mock["raw_text"])
        if prompt_type == 'prompt_quickstart':
            base_details["resource_list"] = []
        else:
            base_details["resource_list"] = filter_documents(
                mock["raw_resources"])

        return base_details


"""
prompt_quickstart empty resource_list
prompt_developer populate resource_list
prompt_academia populate resource_list
"""


def base_chat_generate(prompt, taskDescription, conversation_history):
    base_history = [
        {
            "role": "USER", "message": f"Here is my base task description: {taskDescription}. You are a helpful assistant whose role is to answer anything from me that is related to the task description."
        },
        {
            "role": "CHATBOT", "message": "Yes, I understand. I am ready to help you with anything you say."
        }
    ]
    base_history.extend(conversation_history)

    if call_api:
        ...
        if co is None:
            return ("key not found")

        response = co.chat(
            model='command-nightly',
            message=prompt,
            temperature=0.7,
            chat_history=base_history,
        )
        raw_response = response.text
    else:
        ...
        mock = mockdata()
        raw_response = mock["chat_response"]
        time.sleep(3)

    return raw_response


# TODO fine tune prompt


def prompt_quickstart(input):
    ...
    if co is None:
        return ("key not found")

    response = co.chat(
        model='command-nightly',
        message=f"here is my description: {input}. Generate a numbered list of instructions to achieve my goal.",
        temperature=0.7,
        chat_history=[
            {
                "role": "USER", "message": "I need you to generate a numbered list of instructions to achieve my goal."
            },
            {
                "role": "CHATBOT", "message": "Sure, provide me with your goal, and my task is to create a numbered list of instructions."
            }
        ],
    )
    return response


def prompt_developer(input):
    ...
    if co is None:
        return ("key not found")

    response = co.chat(
        model='command-nightly',
        message=f"here is my description: {input}. Generate a numbered list of instructions to achieve my goal.",
        connectors=[{"id": "web-search"}],
        temperature=0,
        chat_history=[
            {
                "role": "USER", "message": "I need you to generate a numbered list of instructions to achieve my goal."
            },
            {
                "role": "CHATBOT", "message": "Sure, provide me with your goal, and my task is to create a numbered list of instructions."
            }
        ],
    )

    return response


def prompt_academia(input):
    ...
    if co is None:
        return ("key not found")

    response = co.chat(
        model='command-nightly',
        message=f"here is my description: {input}. Generate a numbered list of instructions to achieve my goal.",
        connectors=[{"id": "web-search"}],
        temperature=0,
        chat_history=[
            {
                "role": "USER", "message": "I need you to generate a numbered list of instructions to achieve my goal."
            },
            {
                "role": "CHATBOT", "message": "Sure, provide me with your goal, and my task is to create a numbered list of instructions."
            }
        ],
    )

    return response
