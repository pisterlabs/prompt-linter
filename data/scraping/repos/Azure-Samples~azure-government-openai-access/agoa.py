#Note: The openai-python library support for Azure OpenAI is in preview.

import os
import time
import openai
openai.api_type = "azure"

if (os.getenv("OPENAI_API_BASE") == None) :
  print("Please set the OPENAI_API_BASE environment variable to the API base URL for Azure OpenAI.")
  exit(1)
if (os.getenv("OPENAI_API_KEY") == None) :
  print("Please set the OPENAI_API_KEY environment variable to the API key for Azure OpenAI.")
  exit(1)
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2022-12-01"
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = os.getenv("OPENAI_ENGINE","text-davinci-003")
os.system('cls' if os.name == 'nt' else 'clear')
while True:
  
  print()

  response = openai.Completion.create(engine=engine, prompt="Write a tagline for an ice cream shop.", max_tokens=100)

  print ( str(response['choices'][0]['text']) )

  time.sleep(5) 