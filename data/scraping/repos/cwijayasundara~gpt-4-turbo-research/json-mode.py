import openai
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

print(openai.api_key)

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={"type": "json_object"},
  messages=[
    {"role": "system", "content": "You are a helpful programmer who always returns your answer in JSON."},
    {"role": "user", "content": "give me a list of 5 things for grocery shopping. call the list 'groceries'"}
  ]
)

print(completion.choices[0].message)

groceries_list = json.loads(completion.choices[0].message.content)
print(groceries_list)
print(groceries_list['groceries'][4])