import os

import openai
import tiktoken
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("key")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


text = """
You should express what you want a model to do by \ 
providing instructions that are as clear and \
specidifc as you can possible make them. \
This will guide the model towards the desired output, \
and reduce the chances of receving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short propmt. \
In many cases, longer prompts provide more clarity \
and context for the model, which can lead to \
more detailed and relevant outputs.. 
"""

prompt = f"""
Summarize the text delimited by triple brackets \
into a single sentence.
```{text} ```
"""
response = get_completion(prompt)
print(response)
