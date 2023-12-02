import os
import openai
openai.api_key = ""
response = openai.Image.create(
  prompt=input("prompt: "),
  n=2,
  size="1024x1024"
)

print(response)
