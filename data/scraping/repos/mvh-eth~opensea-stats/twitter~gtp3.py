import os
import re
import openai

openai.api_key = "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"


def request_davinci(prompt, max_tokens):
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].text
