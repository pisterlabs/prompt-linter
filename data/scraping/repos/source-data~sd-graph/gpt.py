"""Utilities for working with OpenAI's GPT-3 API."""

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from . import OPENAI_API_TOKEN

openai.api_key = OPENAI_API_TOKEN


def review_summarization_prompt(reviews_text):
    return [
        {"role": "system", "content": "You are a neutral summarizer."},
        {
            "role": "user",
            "content": (
                "Please synthesize and summarize the reviews below to highlight the"
                " strengths and weaknesses noted in the reviews. The summary should be"
                " succinct and written as a single paragraph."
            ),
        },
        {"role": "user", "content": reviews_text},
    ]


review_summarization_parameters = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["[END]"],
}


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=100),
    reraise=True,
    stop=stop_after_attempt(3),
)
def chat(messages, parameters, dry_run=True):
    """
    Get a chat completion for a prompt from OpenAI's API.

    Arguments:
        prompt: The prompt to complete.
        parameters: A dict of parameters to pass to the OpenAI API.
        dry_run: If True, no API call is made and a message indicating it is returned.
    """
    if dry_run:
        return "This is a dry run. No API call was made."

    completion = openai.ChatCompletion.create(messages=messages, **parameters)
    return completion.choices[0].message.content
