import openai
import os
from string import Template


def get_apikey():
    return os.getenv('OPENAI_KEY')


def format_prompt(prompt, text) -> str:
    if '$text' in prompt:
        t = Template(prompt)
        return t.substitute(text=text)
    else:
        return prompt + text


def completion(prompt='', text='', temperature=0.7, model='ext-davinci-003', max_tokens=1000):
    p = format_prompt(prompt, text)
    print(f"model={model}, temperature={temperature}\n{p}")
    try:
        for ev in openai.Completion.create(
            model=model,
            prompt=p,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        ):
            yield ev['choices'][0]['text']
    except openai.error.OpenAIError as e:
        raise RuntimeError(f"{e=}")


def chat_completion(prompt='', text='', temperature=0.7, model='gpt-3.5-turbo', max_tokens=1000,
                    instruction='Be precise and concise'):
    messages = [
        {'role': 'system', 'content': instruction },  # general instruction to the chat
        {'role': 'user', 'content': format_prompt(prompt, text)}
    ]
    print(f"model={model}, temperature={temperature}\n{messages}")
    try:
        for ev in openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        ):
            if 'content' in ev['choices'][0]['delta']:
                yield ev['choices'][0]['delta']['content']
    except openai.error.OpenAIError as e:
        raise RuntimeError(f"{e=}")


def models():
    try:
        m = openai.Model.list()
        print(m)
    except openai.error.OpenAIError as e:
        raise RuntimeError(f"{e=}")


openai.api_key = get_apikey()
