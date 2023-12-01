import re

import openai

from chats.client_utils import create_client

create_client()


def create_name(full_text: str, model_name:str ):
    prompt = f"""Please give a short title to this document 
    ```
    {full_text}
    ```
    """

    temperature = 0.5
    max_tokens = 250
    args = {
        "model": model_name,

        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    if model_name == "gpt-3.5-turbo-0301":
        args["messages"] = [{"role": "user", "content": prompt}]
    else:
        args["prompt"] = prompt
    response = openai.Completion.create(**args)

    print(response)
    if model_name == "gpt-3.5-turbo-0301":
        title = response["choices"][0]["message"]["content"]
    else:
        title = response["choices"][0]["text"]
    return "".join(re.split("[^a-zA-Z]*", title))
