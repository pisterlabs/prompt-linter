#!/usr/bin/env python3

# Script by Gynvael Coldwind
# https://gynvael.coldwind.pl/?lang=en&id=771

import openai
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# !!!You need an API key!!!
# https://platform.openai.com/account/api-keys
with open(f"/Users/kamilogorek/.openai") as f:
  openai.api_key = f.read().strip()

arg = ' '.join(sys.argv[1:])

r = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  #model="gpt-4",
  messages=[
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."},
        {"role": "user", "content": f"Answer with only the actual command without any intro or explanation. What is the osx command line command to {arg}"}
    ]
)

text = r["choices"][0]["message"]["content"]
if text.startswith('`') and text.endswith('`'):
  text = text[1:-1]

print(text)
