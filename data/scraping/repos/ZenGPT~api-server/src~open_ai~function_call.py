import os
from datetime import datetime

import backoff
import openai
import tiktoken

from open_ai.request_schema import open_ai_schema
from dotenv import load_dotenv

from price_models import openAIModels

load_dotenv()

default_model_name = "gpt-3.5-turbo"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_time=10)
def open_ai_call(options):
    d = open_ai_schema.validate(options)
    openai.api_key = os.getenv('OPEN_AI_KEY')
    return openai.ChatCompletion.create(
        model=d.get('model', default_model_name),
        messages=d.get('prompt', ''),
        temperature=float(d.get('temperature', 1)),
        max_tokens=d.get('max_tokens'),
        top_p=float(d.get('top_p', 1)),
        frequency_penalty=float(d.get('frequency_penalty', 0)),
        presence_penalty=float(d.get('presence_penalty', 0)),
        stop=d.get('stop'),
        n=1,  # d.get('n', 1), we only support 1
        stream=d.get('stream', False),
        user=d.get('user', 'default'),
        timeout=300
    )


def get_prompt(chat_history: [dict], with_chat_ai_prompt: bool = True, length_limit: bool = True) -> ([dict], int):
    """
    get prompt from chat history
    :param length_limit:
    :param with_chat_ai_prompt:
    :param chat_history:
    :return:
    """
    total_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    history = []
    model_info = openAIModels[default_model_name]
    tokens_per_message = model_info['tokens_per_message']
    tokens_per_name = model_info['tokens_per_name']
    for i in reversed(range(len(chat_history))):
        item = chat_history[i]
        new_tokens = tokens_per_message
        if item.get("content") is None: continue  # 'content' is a required property
        if item.get("role") is None: continue  # 'content' is a required property

        for key, value in item.items():
            if key == "name":
                new_tokens += tokens_per_name
            if value is None: continue
            new_tokens += get_token_count(value, default_model_name)
        if length_limit and total_tokens + new_tokens > 3000 and 'gpt-3.5-turbo' in default_model_name:
            break
        if length_limit and total_tokens + new_tokens > 6000 and 'gpt-4' in default_model_name:
            break
        total_tokens += new_tokens
        history.append(item)

    if with_chat_ai_prompt:
        prompt_item = {
            'role': 'system',
            'content': f'You are an AI assistant called "GPTDock" that based on the language model {default_model_name}, '
                       'you are helpful, creative, clever, friendly and honest. '
                       'Every code block must rendered as markdown with the program language name, '
                       'inline code will be wrapped by backtick mark. '
                       'All the formatting must be done by markdown. Render references as normal list with link instead of footnote.'
                       f'Current date and time: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC+0',
        }
        history.append(prompt_item)
        new_tokens = tokens_per_message
        for key, value in prompt_item.items():
            new_tokens += get_token_count(value, default_model_name)
            if key == "name":
                new_tokens += tokens_per_name
        total_tokens += new_tokens

    total_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return list(reversed(history)), total_tokens


def get_request_data(prompt: [dict], prompt_length: int, client_user_id: str, stream: bool) -> dict:
    model_info = openAIModels[default_model_name]
    return {
        "prompt": prompt,
        "temperature": 0.6,
        "max_tokens": model_info['max_tokens'] - prompt_length - 200,  # make sure the completion is not too long
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.4,
        "model": default_model_name,
        "user_token": os.getenv('OPEN_AI_API_KEY'),
        "stream": stream,
        "user": client_user_id
    }


def get_token_count(s: str, model_name: str = default_model_name) -> int:
    return len(tiktoken.encoding_for_model(model_name).encode(s))
