"""Library with OpenAI API solutions as functions

References:

For building code:  https://beta.openai.com/docs/guides/code/introduction

"""

import openai
import os


def submit_question(text):
    """This submits a question to the OpenAI API"""

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = text

    result = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="text-davinci-002",
    )["choices"][0]["text"].strip(" \n")
    return result


# build a function that converts a comment into code in any language
def create_code(text, language):
    """This submits a comment to the OpenAI API to create code in any languag

    Example:
        language = '# Python3'
        text = f"Calculate the mean distance between an array of points"
        create_code(text, language)

    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"## {language}\n\n{text}"

    result = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="davinci-codex",
    )["choices"][0]["text"].strip(" \n")
    return result
