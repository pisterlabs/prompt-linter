import os
import numpy as np

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_response(
    prompt,
    engine="text-davinci-003",
    max_tokens=256,
    temperature=0,
    top_p=1,
    get_logits=False,
):
    response = openai.Completion.create(
        model=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response["choices"][0]["text"]
