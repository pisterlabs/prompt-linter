#!/usr/bin/env python

'''
Model overview: https://platform.openai.com/docs/models/overview
Guide: https://platform.openai.com/docs/models/overview

Example: https://platform.openai.com/docs/models/overview

See: https://github.com/openai/openai-cookbook
More: https://platform.openai.com/examples

https://cobusgreyling.medium.com/example-code-implementation-considerations-for-gpt-3-5-turbo-chatml-whisper-e61f8703c5db
'''

from   pprint import pprint
import openai
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

openai.api_key = OPENAI_API_KEY

messages=[
  {"role": "system", "content": "You are a helpful AI assistant."},
  {"role": "user", "content": "Please tell me about yourself."},
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    max_tokens=200,
    n=1,
    stop=None,
    temperature=0.5,
)

pprint(response)

''' Expected sample output:

<OpenAIObject chat.completion id=chatcmpl-743b1lLxIAlQKv2kGLgRebsnZqkvo at 0x7f0e298cd170> JSON: {
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "I am an artificial intelligence programmed to assist and provide helpful responses to your queries. My purpose is to make your tasks easier and provide you with accurate information. I am constantly learning and improving my abilities to better serve you.",
        "role": "assistant"
      }
    }
  ],
  "created": 1681201491,
  "id": "chatcmpl-743b1lLxIAlQKv2kGLgRebsnZqkvo",
  "model": "gpt-3.5-turbo-0301",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 44,
    "prompt_tokens": 26,
    "total_tokens": 70
  }
'''
