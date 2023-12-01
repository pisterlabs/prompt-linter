#!/usr/bin/env python3

"""An open AI api key is required to run this code. The engine leverages the GPT-3 model, and the code was 
written by Github copilot.
"""

import openai
import os
import click

# write a function that takes a text string and returns a summary of the text
def summarize(text):
    """Summarize a text string and return a summary of the text"""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip(" \n")


@click.command()
@click.argument("text")
def main(text):
    """Summarize a text string and return a summary of the text"""
    print(summarize(text))

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()