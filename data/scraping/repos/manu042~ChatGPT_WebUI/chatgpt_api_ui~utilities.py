import openai
import logging

from django.conf import settings


logger = logging.getLogger('chatgpt_webui')


def chat_completion(messages, temperature, presence_penalty, frequency_penalty):
    # https://platform.openai.com/docs/guides/chat/introduction
    try:
        openai.api_key = settings.CHAT_GPT_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=float(temperature),
            presence_penalty=float(presence_penalty),
            frequency_penalty=float(frequency_penalty)
        )
    except Exception as e:
        raise e

    return response
