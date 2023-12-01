#####################
'''

Follow the instructions in the text file and run main.py

Enjoy!

To be fleshed out later.

'''
#####################


import os
import openai
import gradio as gr

with open("openai_api_key.txt", "r") as f:
    apiKey = f.read()
    openai.api_key = apiKey

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="hello!",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

text = response['choices'][0]['text']
print(text)


