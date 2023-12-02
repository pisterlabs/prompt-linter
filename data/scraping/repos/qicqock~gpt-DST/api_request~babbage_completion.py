import openai
from config import CONFIG

"""GPT3.5 completion"""

def babbage_completion(prompt_text):
    openai.api_key = CONFIG['openai_api_key']
    return openai.Completion.create(
        engine='text-babbage-001',
        prompt=prompt_text,
        # max_tokens=1500,
        temperature=0,
        stop=['--', '\n', ';', '#'],
    )["choices"][0]["text"]