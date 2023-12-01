"""
Advanced prompt engineering approaches without context. 
"""

import openai
import os

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Zero Shot Prompting
# PROMPT_FILE_PATH = "./prompts/vanilla-prompt-zero-shot.txt"
# TOPIC = "Feed Forward Neural Networks in NLP using PyTorch"
# RESPONSE_FILE_PATH = lambda timestamp: f"./responsebuffers/{timestamp} (vanilla-zero-shot).txt"

# prompt = str()
# with open(PROMPT_FILE_PATH, "r+") as file:
#   prompt = file.read()
#   prompt = prompt.replace("<<TOPIC>>", TOPIC)

# Few Shot Prompting
PROMPT_PREFACE_FILE_PATH = "./prompts/vaniila-prompt-few-shot-preface.txt"
TOPIC = "Feed Forward Neural Networks in NLP using PyTorch"
PROMPT_EXAMPLES_FILE_PATH = "./prompts/few-shot-examples.txt"
RESPONSE_FILE_PATH = lambda timestamp: f"./responsebuffers/{timestamp} (vanilla-few-shot).txt"

prompt = str()
with open(PROMPT_PREFACE_FILE_PATH, "r+") as file:
  prompt = file.read()
  prompt = prompt.replace("<<TOPIC>>", TOPIC)

with open(PROMPT_EXAMPLES_FILE_PATH, "r+") as file:
  prompt += file.read()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORGANIZATON"))

completion = client.chat.completions.create(
  model="gpt-4-1106-preview",
  messages=[
    {
      "role": "system",
      "content": f"You are a highly skilled subject matter expert in {TOPIC} and a creative question architect.",
    },
    {
      "role": "user", 
      "content": prompt
    },
  ],
  temperature=0.8,
  max_tokens=4096,
)

response = completion.choices[0].message.content

with open(RESPONSE_FILE_PATH(datetime.now().strftime("%d-%m-%Y %H:%M:%S")), "w+") as file:
  file.write(response)

