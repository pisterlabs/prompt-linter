import os
import openai
from config import apikey

openai.api_key = apikey

response = openai.ChatCompletion.create(
  engine="davinci-codex",
  messages=[
    {
      "role": "system",
      "content": "write a letter to boss for salary increament"
    },
    {
      "role": "user",
      "content": ""
    },
    {
      "role": "assistant"
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

