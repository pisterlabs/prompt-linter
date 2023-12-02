import random
import os
import openai


model="text-davinci-003"
max_tokens=256
temperature=1
n = 1

OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY


def Davinci_003_openai(prompt):
    response = openai.Completion.create(
    model=model,
    prompt = prompt, 
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    logprobs = 5)
    return response