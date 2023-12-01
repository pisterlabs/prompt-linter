import os
import openai
import re

# load environment variables 
from dotenv import load_dotenv
load_dotenv()

# system config
openai.api_key = os.getenv("AZURE_KEY")
openai.api_base = os.getenv("AZURE_ENDPOINT")
openai.api_type = os.getenv("API_TYPE")
openai.api_version = os.getenv("API_VERSION")

model = os.getenv("AZURE_MODEL")

# gpt config
temperature = 0.7
max_tokens = 1000


async def getMemeContent(prompt):
  prompt_templete = f"""
    generate a meme about {prompt}.

    with one sentence of top caption and one sentence of bottom caption and a image prompt used by midjourney.
    Reply to me according to the following format, do not say anything else.

    TOP: one sentence of top caption, 
    BOTTOM: one sentence of bottom caption, 
    PROMPT: a image describe used by midjourney
    """

  messages = [
    {
      "role": "assistant",
      "content": prompt_templete
    },
  ]

  # print(prompt_templete)

  response = openai.ChatCompletion.create(engine=model,
                                          messages=messages,
                                          max_tokens=max_tokens,
                                          temperature=temperature)
  result = response.choices[0].message.content
  values = extract_values(result)
  return values


def extract_values(string):
  pattern = r'TOP: (.*?)\nBOTTOM: (.*?)\nPROMPT: (.*?)$'
  match = re.search(pattern, string, re.MULTILINE | re.DOTALL)

  if match:
    top_value = match.group(1)
    bottom_value = match.group(2)
    prompt_value = match.group(3)

    return top_value, bottom_value, prompt_value
  else:
    return None
