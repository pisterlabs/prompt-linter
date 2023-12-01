import os
import openai

from enviroment import environ

env = environ(".env")

openai.organization = env["OPENAI_ORGANIZATION"]
openai.api_key = env["OPENAI_API_KEY"]

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Translate this into 1. French and 2. Spanish\n\nWhat rooms do you have available?\n\n1.",
  temperature=0.3,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)